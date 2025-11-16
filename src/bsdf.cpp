#include "rdr/bsdf.h"

#include "rdr/fresnel.h"
#include "rdr/interaction.h"
#include "rdr/math_aliases.h"
#include "rdr/platform.h"

RDR_NAMESPACE_BEGIN

static Vec3f obtainOrientedNormal(const SurfaceInteraction &interaction,
                                  bool twosided) {
  AssertAllValid(interaction.shading.n);
  AssertAllNormalized(interaction.shading.n);
  return twosided && interaction.cosThetaO() < 0 ? -interaction.shading.n
                                                 : interaction.shading.n;
}

/* ===================================================================== *
 *
 * IdealDiffusion
 *
 * ===================================================================== */

void IdealDiffusion::crossConfiguration(
    const CrossConfigurationContext &context) {
  auto texture_name = properties.getProperty<std::string>("texture_name");
  auto texture_ptr = context.textures.find(texture_name);
  if (texture_ptr != context.textures.end()) {
    texture = texture_ptr->second;
  } else {
    Exception_("Texture [ {} ] not found", texture_name);
  }

  clearProperties();
}

Vec3f IdealDiffusion::evaluate(SurfaceInteraction &interaction) const {
  const Vec3f normal = obtainOrientedNormal(interaction, twosided);
  if (Dot(interaction.wi, normal) < 0 || Dot(interaction.wo, normal) < 0)
    return {0, 0, 0};
  return texture->evaluate(interaction) * INV_PI;
}

Float IdealDiffusion::pdf(SurfaceInteraction &interaction) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f IdealDiffusion::sample(SurfaceInteraction &interaction, Sampler &sampler,
                             Float *out_pdf) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

/// return whether the bsdf is perfect transparent or perfect reflection
bool IdealDiffusion::isDelta() const { return false; }

/* ===================================================================== *
 *
 * PerfectRefraction
 *
 * ===================================================================== */

PerfectRefraction::PerfectRefraction(const Properties &props)
    : eta(props.getProperty<Float>("eta", 1.5F)), BSDF(props) {}

Vec3f PerfectRefraction::evaluate(SurfaceInteraction &) const {
  // Since this is a delta distribution, it has no contribution to the queried
  // direction
  return {0.0, 0.0, 0.0};
}

Float PerfectRefraction::pdf(SurfaceInteraction &) const { return 0; }

Vec3f PerfectRefraction::sample(SurfaceInteraction &interaction,
                                Sampler &sampler, Float *pdf) const {
  // The interface normal
  Vec3f normal = interaction.shading.n;
  // Cosine of the incident angle
  Float cos_theta_i = Dot(normal, interaction.wo);
  // Whether the ray is entering the medium
  bool entering = cos_theta_i > 0;
  // Corrected eta by direction
  Float eta_corrected = entering ? eta : 1.0F / eta;

  if (!Refract(interaction.wo, normal, eta_corrected, interaction.wi)) {
    interaction.wi = Reflect(interaction.wo, normal);
  }

  // Set the pdf and return value, we dont need to understand the value now
  if (pdf != nullptr)
    *pdf = 1.0F;
  return Vec3f(1.0);
}

bool PerfectRefraction::isDelta() const { return true; }

/* ===================================================================== *
 *
 * FresnelSpecular
 *
 * ===================================================================== */

Glass::Glass(const Properties &props)
    : R(props.getProperty<Vec3f>("R", Vec3f(1.0))),
      T(props.getProperty<Vec3f>("T", Vec3f(1.0))),
      eta(props.getProperty<Float>("eta", 1.5F)), BSDF(props) {}

Vec3f Glass::evaluate(SurfaceInteraction &) const {
  // Since this is a delta distribution, it has no contribution to the queried
  // direction
  return {0.0, 0.0, 0.0};
}

Float Glass::pdf(SurfaceInteraction &) const { return 0; }

Vec3f Glass::sample(SurfaceInteraction &interaction, Sampler &sampler,
                    Float *pdf) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

bool Glass::isDelta() const { return true; }

/* ===================================================================== *
 *
 * MicrofacetReflection
 *
 * ===================================================================== */

void MicrofacetReflection::crossConfiguration(
    const CrossConfigurationContext &context) {
  auto texture_name = properties.getProperty<std::string>("texture_name");
  auto texture_ptr = context.textures.find(texture_name);
  if (texture_ptr != context.textures.end()) {
    R = texture_ptr->second;
  } else {
    Exception_("Texture {} not found", texture_name);
  }

  clearProperties();
}

MicrofacetReflection::MicrofacetReflection(const Properties &props)
    : k(props.getProperty<Vec3f>("k", Vec3f(1.0))),
      etaI(props.getProperty<Vec3f>("etaI", Vec3f(1.0F))),
      etaT(props.getProperty<Vec3f>("etaT", Vec3f(1.0F))),
      dist(props.getProperty<Float>("alpha_x", 0.1),
           props.getProperty<Float>("alpha_y", 0.1)),
      BSDF(props) {}

Vec3f MicrofacetReflection::evaluate(SurfaceInteraction &interaction) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Float MicrofacetReflection::pdf(SurfaceInteraction &interaction) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f MicrofacetReflection::sample(SurfaceInteraction &interaction,
                                   Sampler &sampler, Float *pdf_in) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

/// return whether the bsdf is perfect transparent or perfect reflection
bool MicrofacetReflection::isDelta() const { return false; }

RDR_NAMESPACE_END
