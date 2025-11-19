#include "rdr/integrator.h"

#include <omp.h>

#include "rdr/bsdf.h"
#include "rdr/camera.h"
#include "rdr/canary.h"
#include "rdr/film.h"
#include "rdr/halton.h"
#include "rdr/interaction.h"
#include "rdr/light.h"
#include "rdr/math_aliases.h"
#include "rdr/math_utils.h"
#include "rdr/platform.h"
#include "rdr/properties.h"
#include "rdr/ray.h"
#include "rdr/scene.h"
#include "rdr/sdtree.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * Intersection Test Integrator's Implementation
 *
 * ===================================================================== */

void IntersectionTestIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // Statistics
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      for (int sample = 0; sample < spp; sample++) {
        const Vec2f &pixel_sample = sampler.getPixelSample();
        auto ray =
            camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);

        // Accumulate radiance
        assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        const Vec3f &L = Li(scene, ray, sampler);
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f IntersectionTestIntegrator::Li(ref<Scene> scene, DifferentialRay &ray,
                                     Sampler &sampler) const {
  Vec3f color(0.0);

  // Cast a ray until we hit a non-specular surface or miss
  // Record whether we have found a diffuse surface
  bool diffuse_found = false;
  SurfaceInteraction interaction;

  for (int i = 0; i < max_depth; ++i) {
    interaction = SurfaceInteraction();
    bool intersected = scene->intersect(ray, interaction);

    // Perform RTTI to determine the type of the surface
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(interaction.bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(interaction.bsdf) != nullptr;

    // Set the outgoing direction
    interaction.wo = -ray.direction;

    if (!intersected) {
      break;
    }

    // Check if the intersection point is within the emissive rectangle
    if (area_light_positions.size() == 4) {
      const Vec3f &p = interaction.p;

      // Create plane equation from three points of the rectangle
      const Vec3f &a = area_light_positions[0];
      const Vec3f &b = area_light_positions[1];
      const Vec3f &c = area_light_positions[2];

      // Calculate normal of the rectangle plane
      Vec3f normal = Normalize(Cross(b - a, c - a));

      // Check if point lies on the plane
      Float distance = Dot(p - a, normal);
      if (std::abs(distance) < 1e-4f) {
        // Project point onto 2D coordinate system of the rectangle
        Vec3f u_axis = Normalize(b - a);
        Vec3f v_axis = Normalize(c - b);

        Vec3f local_p = p - a;
        Float u_coord = Dot(local_p, u_axis);
        Float v_coord = Dot(local_p, v_axis);

        // Check if point is within bounds of the rectangle
        Float u_length = Norm(b - a);
        Float v_length = Norm(c - b);

        if (u_coord >= 0 && u_coord <= u_length && v_coord >= 0 &&
            v_coord <= v_length) {
          // Point is within the emissive rectangle
          return area_light_flux;
        }
      }
    }

    if (is_perfect_refraction) {
      // We should follow the specular direction
      Float *pdf;
      Vec3f f = interaction.bsdf->sample(interaction, sampler, pdf);
      ray = interaction.spawnRay(interaction.wi);
      continue;
    }

    if (is_ideal_diffuse) {
      // We only consider diffuse surfaces for direct lighting
      diffuse_found = true;
      break;
    }

    // We simply omit any other types of surfaces
    break;
  }

  if (!diffuse_found) {
    return color;
  }

  color = directLighting(scene, interaction);
  return color;
}

Vec3f IntersectionTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction) const {
  Vec3f color(0, 0, 0);

  // Extract common lighting calculation function for point lights
  auto computePointLightContribution = [&](const Vec3f &light_pos,
                                           const Vec3f &light_flux) -> Vec3f {
    Float dist_to_light = Norm(light_pos - interaction.p);
    Vec3f light_dir = Normalize(light_pos - interaction.p);
    auto test_ray = DifferentialRay(interaction.p, light_dir);

    SurfaceInteraction shadow_interaction;
    if (scene->intersect(test_ray, shadow_interaction)) {
      if (Norm(shadow_interaction.p - interaction.p) < dist_to_light - 1e-4f) {
        return Vec3f(0, 0, 0); // Occluded
      }
    }

    // Not occluded, calculate contribution
    const BSDF *bsdf = interaction.bsdf;
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

    if (bsdf != nullptr && is_ideal_diffuse) {
      Float cos_theta = std::max(Dot(light_dir, interaction.normal), 0.0f);
      return bsdf->evaluate(interaction) * cos_theta * light_flux /
             (4.0f * PI * dist_to_light * dist_to_light);
    }

    return Vec3f(0, 0, 0);
  };

  // Calculate contributions from both point light sources
  color +=
      computePointLightContribution(point_light_position1, point_light_flux1);
  color +=
      computePointLightContribution(point_light_position2, point_light_flux2);

  // Add area light contribution if it exists
  if (area_light_positions.size() == 4) {
    const int area_light_samples = 256; // Number of samples for soft shadows
    Vec3f area_light_contribution(0.0f);

    for (int i = 0; i < area_light_samples; i++) {
      // Sample a point on the rectangular area light
      int a = i / 16;
      int b = i % 16;
      Vec3f light_point =
          Vec3f(-0.46875f + a * 0.0625f, 2.0f, -0.46875f + b * 0.0625f);

      // Calculate light flux (assuming uniform distribution)
      Vec3f light_radiance = area_light_flux / PI;

      // Calculate direction to light
      Float dist_to_light = Norm(light_point - interaction.p);
      Vec3f light_dir = Normalize(light_point - interaction.p);
      Vec3f dir_from_light = Normalize(interaction.p - light_point);

      // Check for occlusion
      auto test_ray = DifferentialRay(interaction.p, light_dir);
      SurfaceInteraction shadow_interaction;
      if (scene->intersect(test_ray, shadow_interaction)) {
        if (Norm(shadow_interaction.p - interaction.p) <
            dist_to_light - 1e-4f) {
          continue; // This sample is occluded
        }
      }

      // Not occluded, calculate contribution for this sample
      const BSDF *bsdf = interaction.bsdf;
      bool is_ideal_diffuse =
          dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

      if (bsdf != nullptr && is_ideal_diffuse) {
        Float cos_theta = std::max(Dot(light_dir, interaction.normal), 0.0f);
        Float cos_l =
            std::max(Dot(dir_from_light, Vec3f(0.0f, -1.0f, 0.0f)), 0.0f);
        area_light_contribution += bsdf->evaluate(interaction) * cos_theta *
                                   cos_l * light_radiance /
                                   (dist_to_light * dist_to_light);
      }
    }

    color += area_light_contribution / static_cast<Float>(area_light_samples);
  }

  return color;
}

/* ===================================================================== *
 *
 * Path Integrator's Implementation
 *
 * ===================================================================== */

void PathIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::Li(ref<Scene> scene, DifferentialRay &ray,
                         Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::directLighting(ref<Scene> scene,
                                     SurfaceInteraction &interaction,
                                     Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

/* ===================================================================== *
 *
 * New Integrator's Implementation
 *
 * ===================================================================== */

// Instantiate template
// clang-format off
template Vec3f
IncrementalPathIntegrator::Li<Path>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
template Vec3f
IncrementalPathIntegrator::Li<PathImmediate>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
// clang-format on

// This is exactly a way to separate dec and def
template <typename PathType>
Vec3f IncrementalPathIntegrator::Li( // NOLINT
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

RDR_NAMESPACE_END
