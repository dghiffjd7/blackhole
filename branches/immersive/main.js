const SOLAR_MASS = 1.98847e30;
const G = 6.67430e-11;
const C = 299_792_458;

const defaultState = {
  massSolar: 6.5e8,
  spin: 0.998,
  inclinationDeg: 60,
  azimuthDeg: 0,
  elevationDeg: 359.8,
  zoom: 0.53,
  diskWidth: 12,
  diskTemp: 1.1,
  doppler: 1.5,
};

const toRadians = (deg) => (deg * Math.PI) / 180;
const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
const wrap360 = (deg) => ((deg % 360) + 360) % 360;

function gravitationalRadius(massSolar) {
  return (2 * G * massSolar * SOLAR_MASS) / (C * C);
}

function iscoRadius(spin) {
  const a = clamp(spin, -0.998, 0.998);
  const term = Math.cbrt(1 - a * a);
  const z1 = 1 + term * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
  const z2 = Math.sqrt(3 * a * a + z1 * z1);
  const sign = a >= 0 ? 1 : -1;
  return 3 + z2 - sign * Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

function photonRadius(spin) {
  const a = Math.abs(clamp(spin, -0.999, 0.999));
  const angle = (2 / 3) * Math.acos(-a);
  return 2 * (1 + Math.cos(angle));
}

function gravitationalRedshift(rRg, spin) {
  const r = Math.max(rRg, 1.0001);
  const a = clamp(spin, -0.998, 0.998);
  const gm = 1 - 2 / r + (a * a) / (r * r);
  return 1 / Math.sqrt(Math.max(gm, 1e-6)) - 1;
}

function orbitalBeta(rRg, spin) {
  const r = Math.max(rRg, 1.0001);
  const omega = 1 / (Math.pow(r, 1.5) + spin);
  return clamp((r * omega) / C, -0.99, 0.99);
}

class ImmersiveRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl =
      canvas.getContext("webgl2", { antialias: true, preserveDrawingBuffer: false }) ??
      canvas.getContext("webgl");
    if (!this.gl) {
      throw new Error("需要 WebGL2 支援");
    }
    this.state = {
      spin: defaultState.spin,
      inclination: toRadians(defaultState.inclinationDeg),
      azimuth: toRadians(defaultState.azimuthDeg),
      elevation: toRadians(defaultState.elevationDeg),
      zoom: defaultState.zoom,
      diskWidth: defaultState.diskWidth,
      diskTemp: defaultState.diskTemp,
      doppler: defaultState.doppler,
    };
    this._initProgram();
    this._resize();
    window.addEventListener("resize", () => this._resize());
  }

  _initProgram() {
    const gl = this.gl;
    const vertexSrc = `#version 300 es
      precision highp float;
      layout(location = 0) in vec2 position;
      out vec2 vUv;
      void main() {
        vUv = position * 0.5 + 0.5;
        gl_Position = vec4(position, 0.0, 1.0);
      }`;

    const fragmentSrc = `#version 300 es
      precision highp float;
      in vec2 vUv;
      out vec4 fragColor;

      uniform vec2 u_resolution;
      uniform float u_time;
      uniform float u_spin;
      uniform float u_inclination;
      uniform float u_azimuth;
      uniform float u_elevation;
      uniform float u_zoom;
      uniform float u_diskWidth;
      uniform float u_diskTemp;
      uniform float u_doppler;

      float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
      }

      float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f*f*(3.0-2.0*f);
        return mix(
          mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
          mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x),
          f.y
        );
      }

      mat2 rot(float a) {
        float s = sin(a);
        float c = cos(a);
        return mat2(c, -s, s, c);
      }

      vec2 projectRay(vec2 uv) {
        vec2 scaled = uv / max(u_zoom, 0.2);
        vec3 dir = normalize(vec3(scaled, -1.35));
        mat3 RX = mat3(
          1.0, 0.0, 0.0,
          0.0, cos(u_elevation), -sin(u_elevation),
          0.0, sin(u_elevation), cos(u_elevation)
        );
        mat3 RY = mat3(
          cos(u_azimuth), 0.0, sin(u_azimuth),
          0.0, 1.0, 0.0,
          -sin(u_azimuth), 0.0, cos(u_azimuth)
        );
        vec3 rotated = RY * RX * dir;
        float depth = max(0.2, -rotated.z);
        return rotated.xy / depth;
      }

      void main() {
        vec2 uv = vUv * 2.0 - 1.0;
        uv.x *= u_resolution.x / u_resolution.y;
        vec2 ray = projectRay(uv);
        float radius = length(ray);
        float phi = atan(ray.y, ray.x);

        float lensing = 1.0 / (1.0 + pow(max(radius - 0.24, 0.0) * 3.0, 2.2));
        lensing += 0.4 / (pow(radius + 0.03, 3.0) + 0.12);

        float photon = exp(-pow((radius - 0.32 - 0.05 * u_spin), 2.0) * 190.0);
        photon += 0.25 * exp(-pow((radius - 0.56 + 0.08 * u_spin), 2.0) * 90.0);

        float diskWindow = smoothstep(2.5, 3.0, radius * u_diskWidth) *
                           (1.0 - smoothstep(u_diskWidth - 0.9, u_diskWidth, radius * 20.0));

        float warp = 1.0 / (1.0 + pow(radius * 2.0, 2.1));
        float doppler = 1.0 + u_doppler * u_spin * cos(phi) * sin(u_inclination);
        doppler = clamp(doppler, 0.1, 3.4);
        float temp = warp * doppler;
        vec3 diskColor = mix(vec3(0.05, 0.08, 0.16), vec3(1.1, 0.7, 0.33), temp);
        diskColor += vec3(2.0, 1.2, 0.65) * pow(max(0.0, temp - 0.4), 2.4);
        diskColor *= lensing * diskWindow * u_diskTemp;
        diskColor *= 0.8 + 0.2 * sin(phi * 5.0 - radius * 12.0 + u_time * 1.3);

        vec3 halo = vec3(1.8, 1.2, 0.8) * photon;
        halo += vec3(1.2, 0.6, 0.25) * exp(-pow((ray.y * 5.0 - 1.5), 2.0) - abs(ray.x) * 2.2);

        float starfield = pow(noise(ray * 55.0 + u_time * 0.2), 40.0);
        vec3 background = vec3(0.005, 0.01, 0.025) + vec3(0.12) * starfield;

        float shadow = smoothstep(0.24, 0.33, radius);
        vec3 color = mix(background, diskColor + halo, shadow);

        vec3 tone = vec3(1.0) - exp(-color * (1.1 + 0.4 * photon));
        tone = pow(tone, vec3(0.55));
        fragColor = vec4(tone, 1.0);
      }`;

    this.program = this._createProgram(vertexSrc, fragmentSrc);
    this.buffers = this._createQuad();
    const glUniform = (name) => this.gl.getUniformLocation(this.program, name);
    this.uniforms = {
      resolution: glUniform("u_resolution"),
      time: glUniform("u_time"),
      spin: glUniform("u_spin"),
      inclination: glUniform("u_inclination"),
      azimuth: glUniform("u_azimuth"),
      elevation: glUniform("u_elevation"),
      zoom: glUniform("u_zoom"),
      diskWidth: glUniform("u_diskWidth"),
      diskTemp: glUniform("u_diskTemp"),
      doppler: glUniform("u_doppler"),
    };
  }

  _createProgram(vs, fs) {
    const gl = this.gl;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader));
      }
      return shader;
    };
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
    }
    return program;
  }

  _createQuad() {
    const gl = this.gl;
    const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
    const vao = gl.createVertexArray();
    const vbo = gl.createBuffer();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    return { vao, vbo, count: 6 };
  }

  _resize() {
    const { canvas, gl } = this;
    const ratio = window.devicePixelRatio || 1;
    const displayWidth = Math.floor(canvas.clientWidth * ratio);
    const displayHeight = Math.floor(canvas.clientHeight * ratio);
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
  }

  updateState(patch) {
    this.state = { ...this.state, ...patch };
  }

  render(timeSec) {
    const gl = this.gl;
    gl.useProgram(this.program);
    gl.bindVertexArray(this.buffers.vao);
    gl.uniform2f(this.uniforms.resolution, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.uniform1f(this.uniforms.time, timeSec);
    gl.uniform1f(this.uniforms.spin, this.state.spin);
    gl.uniform1f(this.uniforms.inclination, this.state.inclination);
    gl.uniform1f(this.uniforms.azimuth, this.state.azimuth);
    gl.uniform1f(this.uniforms.elevation, this.state.elevation);
    gl.uniform1f(this.uniforms.zoom, this.state.zoom);
    gl.uniform1f(this.uniforms.diskWidth, this.state.diskWidth);
    gl.uniform1f(this.uniforms.diskTemp, this.state.diskTemp);
    gl.uniform1f(this.uniforms.doppler, this.state.doppler);
    gl.drawArrays(this.gl.TRIANGLES, 0, this.buffers.count);
    gl.bindVertexArray(null);
  }
}

class ImmersiveLab {
  constructor() {
    this.canvas = document.getElementById("immersive-canvas");
    this.panel = document.getElementById("ui-panel");
    this.toggleButton = document.getElementById("toggle-panel");
    this.metrics = {
      rg: document.getElementById("metric-rg"),
      isco: document.getElementById("metric-isco"),
      photon: document.getElementById("metric-photon"),
      redshift: document.getElementById("metric-redshift"),
      beta: document.getElementById("metric-beta"),
    };
    this.derivationPanel = document.getElementById("derivation-panel");
    this.valueLabels = {
      mass: document.querySelector('[data-value="mass"]'),
      spin: document.querySelector('[data-value="spin"]'),
      inclination: document.querySelector('[data-value="inclination"]'),
      azimuth: document.querySelector('[data-value="azimuth"]'),
      elevation: document.querySelector('[data-value="elevation"]'),
      zoom: document.querySelector('[data-value="zoom"]'),
      diskWidth: document.querySelector('[data-value="diskWidth"]'),
      diskTemp: document.querySelector('[data-value="diskTemp"]'),
      doppler: document.querySelector('[data-value="doppler"]'),
    };
    this.controls = {
      mass: document.getElementById("control-mass"),
      spin: document.getElementById("control-spin"),
      inclination: document.getElementById("control-inclination"),
      azimuth: document.getElementById("control-azimuth"),
      elevation: document.getElementById("control-elevation"),
      zoom: document.getElementById("control-zoom"),
      diskWidth: document.getElementById("control-disk-width"),
      diskTemp: document.getElementById("control-disk-temp"),
      doppler: document.getElementById("control-beam"),
    };
    this.state = { ...defaultState };
    this.renderer = new ImmersiveRenderer(this.canvas);
    this._bindControls();
    this._bindButtons();
    this._updateLabels();
    this._updatePhysics();
    this._loop();
  }

  _bindControls() {
    Object.entries(this.controls).forEach(([key, element]) => {
      element.addEventListener("input", (event) => {
        const value = parseFloat(event.target.value);
        switch (key) {
          case "mass":
            this.state.massSolar = value * 1e8;
            break;
          case "inclination":
            this.state.inclinationDeg = value;
            break;
          case "azimuth":
            this.state.azimuthDeg = value;
            break;
          case "elevation":
            this.state.elevationDeg = value;
            break;
          case "diskWidth":
            this.state.diskWidth = value;
            break;
          case "diskTemp":
            this.state.diskTemp = value;
            break;
          case "doppler":
            this.state.doppler = value;
            break;
          case "zoom":
            this.state.zoom = value;
            break;
          default:
            this.state[key] = value;
        }
        this._updateLabels();
        this._updatePhysics();
      });
    });
  }

  _bindButtons() {
    this.toggleButton.addEventListener("click", () => {
      this.panel.classList.toggle("collapsed");
      const hidden = this.panel.classList.contains("collapsed");
      this.panel.style.transform = hidden ? "translateX(-110%)" : "translateX(0)";
      this.toggleButton.textContent = hidden ? "展開控制台" : "收合控制台";
    });
    document.getElementById("btn-reset").addEventListener("click", () => {
      this.state = { ...defaultState };
      this.controls.mass.value = this.state.massSolar / 1e8;
      this.controls.spin.value = this.state.spin;
      this.controls.inclination.value = this.state.inclinationDeg;
      this.controls.azimuth.value = this.state.azimuthDeg;
      this.controls.elevation.value = this.state.elevationDeg;
      this.controls.zoom.value = this.state.zoom;
      this.controls.diskWidth.value = this.state.diskWidth;
      this.controls.diskTemp.value = this.state.diskTemp;
      this.controls.doppler.value = this.state.doppler;
      this._updateLabels();
      this._updatePhysics();
    });
    document.getElementById("btn-sync-main").addEventListener("click", () => {
      window.location.href = "../index.html";
    });
  }

  _updateLabels() {
    const map = [
      ["mass", this.state.massSolar / 1e8, 2],
      ["spin", this.state.spin, 3],
      ["inclination", this.state.inclinationDeg, 1],
      ["azimuth", wrap360(this.state.azimuthDeg), 1],
      ["elevation", wrap360(this.state.elevationDeg), 1],
      ["zoom", this.state.zoom, 2],
      ["diskWidth", this.state.diskWidth, 2],
      ["diskTemp", this.state.diskTemp, 2],
      ["doppler", this.state.doppler, 2],
    ];
    map.forEach(([key, value, digits]) => {
      const target = this.valueLabels[key];
      if (target) target.textContent = Number(value).toFixed(digits);
    });
  }

  _updatePhysics() {
    const massSolar = this.state.massSolar;
    const massKg = massSolar * SOLAR_MASS;
    const rg = gravitationalRadius(massSolar);
    const isco = iscoRadius(this.state.spin);
    const photon = photonRadius(this.state.spin);
    const redshift = gravitationalRedshift(photon + 0.05, this.state.spin);
    const beta = Math.abs(orbitalBeta(isco, this.state.spin));

    this.metrics.rg.textContent = `${rg.toExponential(3)} m`;
    this.metrics.isco.textContent = `${isco.toFixed(3)} Rg`;
    this.metrics.photon.textContent = `${photon.toFixed(3)} Rg`;
    this.metrics.redshift.textContent = redshift.toFixed(4);
    this.metrics.beta.textContent = beta.toFixed(4);

    const lines = [
      `R_g = 2GM/c² = 2 × ${G} × ${massKg.toExponential(3)} / ${C}²`,
      `  → R_g = ${rg.toExponential(4)} m`,
      `r_ISCO = 3 + Z₂ - sign(a)√((3-Z₁)(3+Z₁+2Z₂))`,
      `  where a = ${this.state.spin.toFixed(3)}, result = ${isco.toFixed(4)} Rg`,
      `r_ph = 2[1 + cos(2/3·acos(-|a|))] = ${photon.toFixed(4)} Rg`,
      `z = 1/√(1 - 2/r + a²/r²) - 1, r = r_ph + 0.05 → ${redshift.toFixed(4)}`,
      `β = r·ω/c, ω = 1/(r^{3/2}+a) → β = ${beta.toFixed(4)}`,
    ];
    this.derivationPanel.textContent = lines.join("\n");

    this.renderer.updateState({
      spin: this.state.spin,
      inclination: toRadians(this.state.inclinationDeg),
      azimuth: toRadians(this.state.azimuthDeg ?? defaultState.azimuthDeg),
      elevation: toRadians(this.state.elevationDeg ?? defaultState.elevationDeg),
      zoom: this.state.zoom,
      diskWidth: this.state.diskWidth / 12,
      diskTemp: this.state.diskTemp,
      doppler: this.state.doppler,
    });
  }

  _loop() {
    const step = (timestamp) => {
      this.renderer.render(timestamp * 0.001);
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  try {
    new ImmersiveLab();
  } catch (error) {
    const canvas = document.getElementById("immersive-canvas");
    canvas.insertAdjacentHTML(
      "beforebegin",
      `<p style="color:#fff;padding:1rem;">初始化失敗：${error.message}</p>`
    );
    console.error(error);
  }
});
