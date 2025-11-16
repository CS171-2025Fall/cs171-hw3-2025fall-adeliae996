#include "rdr/accel.h"

#include "rdr/canary.h"
#include "rdr/interaction.h"
#include "rdr/math_aliases.h"
#include "rdr/platform.h"
#include "rdr/shape.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * AABB Implementations
 *
 * ===================================================================== */

bool AABB::isOverlap(const AABB &other) const {
  return ((other.low_bnd[0] >= this->low_bnd[0] &&
           other.low_bnd[0] <= this->upper_bnd[0]) ||
          (this->low_bnd[0] >= other.low_bnd[0] &&
           this->low_bnd[0] <= other.upper_bnd[0])) &&
         ((other.low_bnd[1] >= this->low_bnd[1] &&
           other.low_bnd[1] <= this->upper_bnd[1]) ||
          (this->low_bnd[1] >= other.low_bnd[1] &&
           this->low_bnd[1] <= other.upper_bnd[1])) &&
         ((other.low_bnd[2] >= this->low_bnd[2] &&
           other.low_bnd[2] <= this->upper_bnd[2]) ||
          (this->low_bnd[2] >= other.low_bnd[2] &&
           this->low_bnd[2] <= other.upper_bnd[2]));
}

bool AABB::intersect(const Ray &ray, Float *t_in, Float *t_out) const {
  // Using the slab method for ray-AABB intersection
  Float tmin = ray.t_min;
  Float tmax = ray.t_max;

  // Check intersection with each axis-aligned slab
  for (int i = 0; i < 3; i++) {
    Float inv_dir = ray.safe_inverse_direction[i];
    Float t0 = (low_bnd[i] - ray.origin[i]) * inv_dir;
    Float t1 = (upper_bnd[i] - ray.origin[i]) * inv_dir;

    // Ensure t0 <= t1
    if (inv_dir < 0.0f) {
      std::swap(t0, t1);
    }

    // Update intersection interval
    tmin = Max(tmin, t0);
    tmax = Min(tmax, t1);

    // Early exit if no intersection
    if (tmin > tmax || tmax <= 0.0f) {
      return false;
    }
  }

  // Store the intersection times
  *t_in = tmin;
  *t_out = tmax;

  return true;
}

/* ===================================================================== *
 *
 * Accelerator Implementations
 *
 * ===================================================================== */

bool TriangleIntersect(Ray &ray, const uint32_t &triangle_index,
                       const ref<TriangleMeshResource> &mesh,
                       SurfaceInteraction &interaction) {
  using InternalScalarType = Double;
  using InternalVecType = Vec<InternalScalarType, 3>;

  AssertAllValid(ray.direction, ray.origin);
  AssertAllNormalized(ray.direction);

  const auto &vertices = mesh->vertices;
  const Vec3u v_idx(&mesh->v_indices[3 * triangle_index]);
  assert(v_idx.x < mesh->vertices.size());
  assert(v_idx.y < mesh->vertices.size());
  assert(v_idx.z < mesh->vertices.size());

  InternalVecType dir = Cast<InternalScalarType>(ray.direction);
  InternalVecType v0 = Cast<InternalScalarType>(vertices[v_idx[0]]);
  InternalVecType v1 = Cast<InternalScalarType>(vertices[v_idx[1]]);
  InternalVecType v2 = Cast<InternalScalarType>(vertices[v_idx[2]]);

  // Calculate edge vectors
  InternalVecType e1 = v1 - v0;
  InternalVecType e2 = v2 - v0;
  InternalVecType h = Cross(dir, e2);
  InternalScalarType det = Dot(e1, h);

  // Ray is parallel to triangle
  if (det > -1e-12 && det < 1e-12) {
    return false;
  }

  InternalScalarType inv_det = 1.0 / det;
  InternalVecType s = Cast<InternalScalarType>(ray.origin) - v0;
  InternalScalarType u = inv_det * Dot(s, h);

  if (u < 0.0 || u > 1.0) {
    return false;
  }

  InternalVecType q = Cross(s, e1);
  InternalScalarType v = inv_det * Dot(dir, q);

  if (v < 0.0 || u + v > 1.0) {
    return false;
  }

  // Calculate t
  InternalScalarType t = inv_det * Dot(e2, q);

  // Check if intersection is within ray range
  if (!ray.withinTimeRange(static_cast<Float>(t))) {
    return false;
  }

  // We will reach here if there is an intersection

  CalculateTriangleDifferentials(interaction,
                                 {static_cast<Float>(1 - u - v),
                                  static_cast<Float>(u), static_cast<Float>(v)},
                                 mesh, triangle_index);
  AssertNear(interaction.p, ray(t));
  assert(ray.withinTimeRange(t));
  ray.setTimeMax(t);
  return true;
}

void Accel::setTriangleMesh(const ref<TriangleMeshResource> &mesh) {
  // Build the bounding box
  AABB bound(Vec3f(Float_INF, Float_INF, Float_INF),
             Vec3f(Float_MINUS_INF, Float_MINUS_INF, Float_MINUS_INF));
  for (auto &vertex : mesh->vertices) {
    bound.low_bnd = Min(bound.low_bnd, vertex);
    bound.upper_bnd = Max(bound.upper_bnd, vertex);
  }

  this->mesh = mesh;   // set the pointer
  this->bound = bound; // set the bounding box
}

void Accel::build() {}

AABB Accel::getBound() const { return bound; }

bool Accel::intersect(Ray &ray, SurfaceInteraction &interaction) const {
  bool success = false;
  for (int i = 0; i < mesh->v_indices.size() / 3; i++)
    success |= TriangleIntersect(ray, i, mesh, interaction);
  return success;
}

RDR_NAMESPACE_END
