#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

namespace kerr {

struct RayResult {
  double deflection;
  double travelTime;
  double closestApproach;
  double hitFlag;
};

struct KerrIntegrator {
  double spin;
  double impact;
  double observerRg;

  static double delta(double r, double a) {
    return r * r - 2.0 * r + a * a;
  }

  static double sigma(double r) { return r * r; }

  static double radialPotential(double r, double a, double l) {
    const double term1 = (r * r + a * a) - a * l;
    const double term2 = l - a;
    const double d = delta(r, a);
    return term1 * term1 - d * term2 * term2;
  }

  static double safeSqrt(double value) {
    return value <= 0.0 ? 0.0 : std::sqrt(value);
  }

  struct Derivs {
    double dr;
    double dphi;
    double dt;
  };

  Derivs eval(double r, double sign) const {
    const double a = spin;
    const double l = impact;
    const double sig = sigma(r);
    double d = delta(r, a);
    if (std::fabs(d) < 1e-9) {
      d = d >= 0.0 ? 1e-9 : -1e-9;
    }
    const double R = radialPotential(r, a, l);
    const double dr = sign * safeSqrt(R) / sig;

    const double numeratorPhi = 2.0 * a * r + (sig - 2.0 * r) * l;
    const double dphi = numeratorPhi / (d * sig);

    const double part = (r * r + a * a);
    const double numeratorT = part * (part - a * l) / d + a * (l - a);
    const double dt = numeratorT / sig;

    return {dr, dphi, dt};
  }

  RayResult integrate(double stepSize, int maxSteps) const {
    double r = observerRg;
    double phi = 0.0;
    double t = 0.0;
    double closest = r;
    double sign = -1.0;  // integrate inward

    for (int i = 0; i < maxSteps; ++i) {
      closest = r < closest ? r : closest;
      if (r <= 1.05) {
        return {phi, t, closest, 1.0};
      }

      const Derivs k1 = eval(r, sign);
      const Derivs k2 = eval(r + 0.5 * stepSize * k1.dr, sign);
      const Derivs k3 = eval(r + 0.5 * stepSize * k2.dr, sign);
      const Derivs k4 = eval(r + stepSize * k3.dr, sign);

      const double dr =
          (stepSize / 6.0) * (k1.dr + 2.0 * k2.dr + 2.0 * k3.dr + k4.dr);
      const double dphi =
          (stepSize / 6.0) * (k1.dphi + 2.0 * k2.dphi + 2.0 * k3.dphi + k4.dphi);
      const double dt =
          (stepSize / 6.0) * (k1.dt + 2.0 * k2.dt + 2.0 * k3.dt + k4.dt);

      const double nextR = r + dr;
      if (!std::isfinite(nextR) || nextR > observerRg * 1.5) {
        break;
      }

      r = nextR;
      phi += dphi;
      t += dt;

      if (std::fabs(dr) < 1e-6 && r > observerRg - 1.0) {
        break;
      }
      if (r < 1.1 && std::fabs(dr) < 1e-6) {
        break;
      }
      if (delta(r, spin) <= 0.0) {
        return {phi, t, closest, 1.0};
      }
    }
    return {phi, t, closest, 0.0};
  }
};

inline RayResult trace(double impact, double spin, double observerRg) {
  KerrIntegrator integrator{spin, impact, observerRg};
  return integrator.integrate(0.01, 20000);
}

}  // namespace kerr

struct TraceConfig {
  double impactMin;
  double impactMax;
  double spin;
  double observerRg;
  std::int32_t samples;
};

struct TraceSample {
  double impact;
  double deflection;
  double travelTime;
  double closestApproach;
  double hitDisk;
};

extern "C" {

EMSCRIPTEN_KEEPALIVE
int trace_kerr_bundle(const TraceConfig* cfg, TraceSample* outSamples,
                      int maxSamples) {
  if (!cfg || !outSamples || cfg->samples <= 0) {
    return -1;
  }
  const int count =
      cfg->samples > maxSamples ? maxSamples : static_cast<int>(cfg->samples);
  const double step =
      (cfg->samples == 1)
          ? 0.0
          : (cfg->impactMax - cfg->impactMin) / (cfg->samples - 1);

  for (int i = 0; i < count; ++i) {
    const double impact = cfg->impactMin + step * i;
    const kerr::RayResult r = kerr::trace(impact, cfg->spin, cfg->observerRg);
    outSamples[i] = {impact, r.deflection, r.travelTime, r.closestApproach,
                     r.hitFlag};
  }
  return count;
}

}
