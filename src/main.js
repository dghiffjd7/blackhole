const SOLAR_MASS = 1.98847e30;
const G = 6.67430e-11;
const C = 299_792_458;

const defaultState = {
  massSolar: 6.5e8,
  spin: 0.998,
  inclinationDeg: 60,
  observerRg: 60,
  diskInnerRg: 1.4,
  diskOuterRg: 12,
  dopplerBoost: 1.5,
  exposure: 1.1,
  flowRate: 1.2,
  cameraAzimuthDeg: 0,
  cameraElevationDeg: 359.8,
  cameraZoom: 0.53,
};

const MASS_SLIDER_SCALE = 1_000_000; // slider units represent millions of solar masses

const toRadians = (deg) => (deg * Math.PI) / 180;
const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
const wrap360 = (deg) => ((deg % 360) + 360) % 360;

function gravitationalRadiusMeters(massSolar) {
  return (2 * G * massSolar * SOLAR_MASS) / (C * C);
}

function iscoRadiusRg(spin) {
  const a = Math.min(Math.max(spin, -0.998), 0.998);
  const term = Math.cbrt(1 - a * a);
  const Z1 = 1 + term * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
  const Z2 = Math.sqrt(3 * a * a + Z1 * Z1);
  const sign = a >= 0 ? 1 : -1;
  return 3 + Z2 - sign * Math.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2));
}

function photonRadiusRg(spin) {
  const a = Math.abs(Math.min(Math.max(spin, -0.999), 0.999));
  const angle = (2 / 3) * Math.acos(-a);
  return 2 * (1 + Math.cos(angle));
}

function gravitationalRedshift(rRg, spin) {
  const r = Math.max(rRg, photonRadiusRg(spin) + 0.01);
  const a = Math.min(Math.max(spin, -0.998), 0.998);
  const gm = 1 - (2 / r) + (a * a) / (r * r);
  const denom = Math.sqrt(Math.max(gm, 1e-6));
  return 1 / denom - 1;
}

function orbitalBeta(rRg, spin) {
  const r = Math.max(rRg, photonRadiusRg(spin) + 0.05);
  const omega = 1 / (Math.pow(r, 1.5) + spin);
  const v = r * omega;
  const beta = Math.min(Math.max(v / C, -0.99), 0.99);
  return Math.abs(beta);
}

class KerrRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", {
      antialias: true,
      preserveDrawingBuffer: false,
    });
    if (!this.gl) {
      throw new Error("需要 WebGL2 支援");
    }
    this.state = {
      spin: defaultState.spin,
      inclination: toRadians(defaultState.inclinationDeg),
      observerRg: defaultState.observerRg,
      gravRadius: gravitationalRadiusMeters(defaultState.massSolar),
      solverEnergy: 0.0,
      diskInnerRg: defaultState.diskInnerRg,
      diskOuterRg: defaultState.diskOuterRg,
      dopplerBoost: defaultState.dopplerBoost,
      exposure: defaultState.exposure,
      flowRate: defaultState.flowRate,
      cameraAzimuth: toRadians(defaultState.cameraAzimuthDeg),
      cameraElevation: toRadians(defaultState.cameraElevationDeg),
      cameraZoom: defaultState.cameraZoom,
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
      uniform float u_observerRg;
      uniform float u_gravRadius;
      uniform float u_solverEnergy;
      uniform float u_diskInnerRg;
      uniform float u_diskOuterRg;
      uniform float u_dopplerBoost;
      uniform float u_exposure;
      uniform float u_flowRate;
      uniform float u_cameraAzimuth;
      uniform float u_cameraElevation;
      uniform float u_cameraZoom;

      float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
      }

      float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        return mix(
          mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), f.x),
          mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x),
          f.y
        );
      }

      float fbm(vec2 p) {
        float value = 0.0;
        float amplitude = 0.5;
        for (int i = 0; i < 4; ++i) {
          value += amplitude * noise(p);
          p = mat2(0.8, -0.6, 0.6, 0.8) * p * 1.8;
          amplitude *= 0.5;
        }
        return value;
      }

      mat2 rot(float a) {
        float s = sin(a);
        float c = cos(a);
        return mat2(c, -s, s, c);
      }

      mat3 rotX3(float a) {
        float s = sin(a);
        float c = cos(a);
        return mat3(
          1.0, 0.0, 0.0,
          0.0, c, -s,
          0.0, s, c
        );
      }

      mat3 rotY3(float a) {
        float s = sin(a);
        float c = cos(a);
        return mat3(
          c, 0.0, s,
          0.0, 1.0, 0.0,
          -s, 0.0, c
        );
      }

      vec2 applyCamera(vec2 uv) {
        vec2 scaled = uv / max(u_cameraZoom, 0.2);
        vec3 dir = normalize(vec3(scaled, -1.35));
        mat3 camera = rotY3(u_cameraAzimuth) * rotX3(u_cameraElevation);
        vec3 rotated = camera * dir;
        float depth = max(0.2, -rotated.z);
        vec2 projected = rotated.xy / depth;
        projected *= 0.92;
        float lift = 0.23 * sin(u_cameraElevation - 1.047197551);
        projected.y += lift;
        return projected;
      }

      vec3 diskColor(float radius, float phi, float spin, float inclination, float dopplerBoost) {
        float warp = 1.0 / (1.0 + pow(radius, 2.2));
        float doppler = (1.0 + dopplerBoost * spin * cos(phi) * sin(inclination));
        doppler = clamp(doppler, 0.05, 3.5);
        float temp = warp * doppler;
        float thermal = smoothstep(0.0, 1.2, temp);
        vec3 cool = vec3(0.08, 0.12, 0.24);
        vec3 warm = vec3(1.1, 0.67, 0.32);
        vec3 color = mix(cool, warm, thermal);
        color += vec3(2.3, 1.4, 0.7) * pow(max(0.0, temp - 0.4), 2.5);
        color += vec3(1.2, 0.45, 0.15) * pow(max(0.0, doppler - 1.3), 2.2);
        return color;
      }

      void main() {
        vec2 uv = vUv * 2.0 - 1.0;
        uv.x *= u_resolution.x / u_resolution.y;
        vec2 cameraUv = applyCamera(uv);
        float radius = length(cameraUv);
        float phi = atan(cameraUv.y, cameraUv.x);

        float spin = u_spin;
        float inclination = u_inclination;

        float gravLens = 1.0 / (1.0 + pow(max(radius - 0.22, 0.0) * 2.4, 2.3));
        gravLens += 0.45 / (pow(radius + 0.03, 3.0) + 0.15);

        float photonRing = exp(-pow((radius - 0.32 - 0.05 * spin), 2.0) * 180.0);
        photonRing += 0.35 * exp(-pow((radius - 0.53 + 0.08 * spin), 2.0) * 80.0);
        photonRing += 0.18 * exp(-pow((radius - 0.68 + 0.03 * sin(u_time)), 2.0) * 40.0);

        float scaledRadius = radius * 24.0;
        float diskWindow = smoothstep(u_diskInnerRg, u_diskInnerRg + 0.8, scaledRadius);
        diskWindow *= 1.0 - smoothstep(u_diskOuterRg - 0.8, u_diskOuterRg, scaledRadius);

        vec3 disk = diskColor(radius * 2.0, phi, spin, inclination, u_dopplerBoost);
        vec2 flowUv =
            rot(spin * 2.5 + u_time * 0.2 * u_flowRate) * cameraUv * (5.0 + u_flowRate);
        float turbulence = fbm(flowUv + vec2(u_time * 0.07, spin));
        float caustics = pow(max(0.0, 1.0 - radius * 1.3), 2.0) * (0.5 + 0.5 * turbulence);
        disk *= gravLens * (0.8 + 0.4 * turbulence);
        disk += vec3(0.9, 0.55, 0.2) * caustics;
        disk *= diskWindow;

        float swirl =
            sin(phi * (3.5 + 0.5 * u_flowRate) + radius * (15.0 + 3.0 * u_flowRate) -
                u_time * (1.2 + 0.6 * spin * u_flowRate));
        disk *= 0.85 + 0.25 * swirl;
        vec2 shearUv = rot(spin * 6.2831 * radius * u_flowRate) * cameraUv;
        float shearHighlights = pow(max(0.0, shearUv.x), 2.5) * exp(-radius * (5.5 - u_flowRate * 0.8));
        disk += vec3(1.2, 0.7, 0.4) * shearHighlights;
        float helix = sin(phi * (6.0 + u_flowRate) - radius * (24.0 - 5.0 * spin) + u_time * 2.4);
        vec3 helixColor = vec3(1.6, 0.9, 0.5) * pow(max(0.0, helix), 3.0) * exp(-radius * 3.5);
        disk += helixColor;

        vec3 photonColor = vec3(2.8, 1.6, 0.85) * photonRing;

        float verticalLens =
            exp(-pow((cameraUv.y * 8.0 - 1.8), 2.0)) * exp(-abs(cameraUv.x) * 1.5);
        float lowerLens =
            exp(-pow((cameraUv.y * 8.0 + 1.4), 2.0)) * exp(-abs(cameraUv.x) * 1.3);
        vec3 halo = vec3(1.6, 1.1, 0.7) * (verticalLens + 0.6 * lowerLens);
        halo *= smoothstep(0.1, 0.5, radius);
        vec3 lensShadow = vec3(0.0);
        float equatorAbsorb = smoothstep(0.0, 0.3, abs(cameraUv.y)) *
                              (1.0 - smoothstep(0.05, 0.25, radius));
        lensShadow -= equatorAbsorb * vec3(0.05, 0.08, 0.12);

        float starfield = smoothstep(0.2, 1.25, radius);
        vec3 background = mix(vec3(0.003, 0.004, 0.012), vec3(0.02, 0.03, 0.05), starfield);
        background += vec3(0.15) *
                      pow(noise(cameraUv * 50.0 + u_time * 0.25) * noise(uv * 35.0), 40.0);
        background *= 0.5;

        float shadow = smoothstep(0.25, 0.35, radius);
        vec3 color = mix(background, disk + photonColor + halo + lensShadow, shadow);

        float solverGlow = smoothstep(0.0, 0.8, u_solverEnergy);
        color += vec3(0.08, 0.25, 0.55) * solverGlow * photonRing;

        float contrast = 1.15 + 0.6 * clamp(u_solverEnergy, 0.0, 1.0);
        vec3 tone = vec3(1.0) - exp(-color * u_exposure * contrast);
        tone = pow(tone, vec3(1.0 / (1.4 + 0.2 * contrast)));

        fragColor = vec4(pow(tone, vec3(1.0 / 1.8)), 1.0);
      }`;

    this.program = this._createProgram(vertexSrc, fragmentSrc);
    this.buffers = this._createScreenQuad();
    this.uniforms = {
      resolution: gl.getUniformLocation(this.program, "u_resolution"),
      time: gl.getUniformLocation(this.program, "u_time"),
      spin: gl.getUniformLocation(this.program, "u_spin"),
      inclination: gl.getUniformLocation(this.program, "u_inclination"),
      observerRg: gl.getUniformLocation(this.program, "u_observerRg"),
      gravRadius: gl.getUniformLocation(this.program, "u_gravRadius"),
      solverEnergy: gl.getUniformLocation(this.program, "u_solverEnergy"),
      diskInner: gl.getUniformLocation(this.program, "u_diskInnerRg"),
      diskOuter: gl.getUniformLocation(this.program, "u_diskOuterRg"),
      dopplerBoost: gl.getUniformLocation(this.program, "u_dopplerBoost"),
      exposure: gl.getUniformLocation(this.program, "u_exposure"),
      flowRate: gl.getUniformLocation(this.program, "u_flowRate"),
      cameraAzimuth: gl.getUniformLocation(this.program, "u_cameraAzimuth"),
      cameraElevation: gl.getUniformLocation(this.program, "u_cameraElevation"),
      cameraZoom: gl.getUniformLocation(this.program, "u_cameraZoom"),
    };
  }

  _createProgram(vertexSrc, fragmentSrc) {
    const gl = this.gl;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new Error(info);
      }
      return shader;
    };
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSrc));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSrc));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
    }
    return program;
  }

  _createScreenQuad() {
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

  pushSolverEnergy(value) {
    this.state.solverEnergy = value;
  }

  render(timeSeconds) {
    const gl = this.gl;
    gl.useProgram(this.program);
    gl.bindVertexArray(this.buffers.vao);

    gl.uniform2f(
      this.uniforms.resolution,
      gl.drawingBufferWidth,
      gl.drawingBufferHeight
    );
    gl.uniform1f(this.uniforms.time, timeSeconds);
    gl.uniform1f(this.uniforms.spin, this.state.spin);
    gl.uniform1f(this.uniforms.inclination, this.state.inclination);
    gl.uniform1f(this.uniforms.observerRg, this.state.observerRg);
    gl.uniform1f(this.uniforms.gravRadius, this.state.gravRadius);
    gl.uniform1f(this.uniforms.solverEnergy, this.state.solverEnergy);
    gl.uniform1f(this.uniforms.diskInner, this.state.diskInnerRg);
    gl.uniform1f(this.uniforms.diskOuter, this.state.diskOuterRg);
    gl.uniform1f(this.uniforms.dopplerBoost, this.state.dopplerBoost);
    gl.uniform1f(this.uniforms.exposure, this.state.exposure);
    gl.uniform1f(this.uniforms.flowRate, this.state.flowRate);
    gl.uniform1f(this.uniforms.cameraAzimuth, this.state.cameraAzimuth);
    gl.uniform1f(this.uniforms.cameraElevation, this.state.cameraElevation);
    gl.uniform1f(this.uniforms.cameraZoom, this.state.cameraZoom);

    gl.drawArrays(gl.TRIANGLES, 0, this.buffers.count);
    gl.bindVertexArray(null);
  }
}

class BlackHoleCard {
  constructor() {
    this.canvas = document.getElementById("bh-canvas");
    this.statusPill = document.getElementById("solver-status");
    this.resetButton = document.getElementById("control-reset");
    this.calcReadout = document.getElementById("calc-readout");
    this.tabButtons = Array.from(
      document.querySelectorAll("[data-tab-target]")
    );
    this.tabPanels = {
      controls: document.getElementById("tab-controls"),
      calculations: document.getElementById("tab-calculations"),
    };
    this.controlValues = {
      mass: document.querySelector('[data-control-value="mass"]'),
      spin: document.querySelector('[data-control-value="spin"]'),
      inclination: document.querySelector('[data-control-value="inclination"]'),
      distance: document.querySelector('[data-control-value="distance"]'),
      diskInner: document.querySelector('[data-control-value="diskInner"]'),
      diskOuter: document.querySelector('[data-control-value="diskOuter"]'),
      doppler: document.querySelector('[data-control-value="doppler"]'),
      exposure: document.querySelector('[data-control-value="exposure"]'),
      flow: document.querySelector('[data-control-value="flow"]'),
      camAzimuth: document.querySelector('[data-control-value="camAzimuth"]'),
      camElevation: document.querySelector('[data-control-value="camElevation"]'),
      camZoom: document.querySelector('[data-control-value="camZoom"]'),
    };
    this.metrics = {
      mass: document.getElementById("metric-mass"),
      spin: document.getElementById("metric-spin"),
      isco: document.getElementById("metric-isco"),
      photon: document.getElementById("metric-photon"),
      redshift: document.getElementById("metric-redshift"),
      fps: document.getElementById("metric-fps"),
    };
    this.controls = {
      mass: document.getElementById("control-mass"),
      spin: document.getElementById("control-spin"),
      inclination: document.getElementById("control-inclination"),
      distance: document.getElementById("control-distance"),
      diskInner: document.getElementById("control-disk-inner"),
      diskOuter: document.getElementById("control-disk-outer"),
      doppler: document.getElementById("control-doppler"),
      exposure: document.getElementById("control-exposure"),
      flow: document.getElementById("control-flow"),
      cameraAzimuth: document.getElementById("control-camera-azimuth"),
      cameraElevation: document.getElementById("control-camera-elevation"),
      cameraZoom: document.getElementById("control-camera-zoom"),
    };
    this.formulaPanel = document.getElementById("formula-panel");
    this.formulaInput = document.getElementById("formula-input");
    this.formulaOutput = document.getElementById("formula-output");
    this.suggestionList = document.getElementById("suggestion-list");
    this.calcConsole = document.getElementById("calc-console");
    this.consoleToggle = document.getElementById("console-toggle");
    this.consoleFullscreen = document.getElementById("console-fullscreen");
    this.consoleInfo = document.getElementById("console-info");
    this.consoleTooltip = document.getElementById("console-tooltip");
    this.applyFormulaBtn = document.getElementById("apply-formula");
    this.clearFormulaBtn = document.getElementById("clear-formula");
    this.subtitleText = document.getElementById("subtitle-text");
    this.overlay = document.getElementById("start-overlay");
    this.initButton = document.getElementById("init-btn");
    this.audioButtons = {
      intro: document.getElementById("narrate-intro"),
      lensing: document.getElementById("narrate-lensing"),
      horizon: document.getElementById("narrate-horizon"),
    };
    this.ambienceToggle = document.getElementById("toggle-ambience");

    this.state = { ...defaultState };
    this.renderer = new KerrRenderer(this.canvas);
    this.lastFrame = performance.now();
    this.frame = 0;
    this.accumTime = 0;
    this.statusPill.textContent = "Shader 近似積分 (待 WASM)";
    this.lastSolverTick = 0;
    this.consoleLive = true;
    this.latestTelemetry = null;
    this.audioCtx = null;
    this.droneGain = null;
    this.isAmbienceOn = true;
    this.narrationClips = this._createNarrationClips();
    this.formulaSuggestions = [
      { label: "z = 1/sqrt(1 - 2/r + a^2/r^2) - 1", insert: "z = 1/sqrt(1 - 2/r + a^2/r^2) - 1" },
      { label: "doppler = sqrt((1+beta)/(1-beta))", insert: "doppler = sqrt((1+beta)/(1-beta))" },
      { label: "inclination = atan(beta) * 180 / PI", insert: "inclination = atan(beta) * 180 / PI" },
      { label: "observer = 45", insert: "observer = 45" },
      { label: "spin = min(0.998, spin + 0.01)", insert: "spin = min(0.998, spin + 0.01)" },
      { label: "mass = 8e8", insert: "mass = 8e8" },
    ];
    this.subtitleTimer = null;

    this._wireControls();
    this._bindReset();
    this._bindTabs();
    this._syncInputsFromState();
    this._initInteractions();
    this._syncCameraControls();
    this._updateTelemetry();
    this._pollValidation();
    this._bindIntroOverlay();
    this._bindAudioControls();
    this._initFormulaConsole();

    this.validationTimer = window.setInterval(
      () => this._pollValidation(),
      10000
    );
    requestAnimationFrame((t) => this._loop(t));
  }

  _wireControls() {
    if (this.controls.mass) {
      this.controls.mass.addEventListener("input", (event) => {
        const millions = parseFloat(event.target.value);
        this.state.massSolar = millions * MASS_SLIDER_SCALE;
        this._updateTelemetry();
      });
    }
    this.controls.spin.addEventListener("input", (event) => {
      const spin = parseFloat(event.target.value);
      this.state.spin = spin;
      this._updateTelemetry();
    });
    this.controls.inclination.addEventListener("input", (event) => {
      const incl = parseFloat(event.target.value);
      this.state.inclinationDeg = incl;
      this._updateTelemetry();
    });
    this.controls.distance.addEventListener("input", (event) => {
      const dist = parseFloat(event.target.value);
      this.state.observerRg = dist;
      this._updateTelemetry();
    });
    this.controls.diskInner.addEventListener("input", (event) => {
      const inner = parseFloat(event.target.value);
      this.state.diskInnerRg = inner;
      if (this.state.diskOuterRg <= inner) {
        this.state.diskOuterRg = inner + 0.5;
        this.controls.diskOuter.value = this.state.diskOuterRg.toString();
      }
      this._updateTelemetry();
    });
    this.controls.diskOuter.addEventListener("input", (event) => {
      const outer = parseFloat(event.target.value);
      this.state.diskOuterRg = Math.max(outer, this.state.diskInnerRg + 0.2);
      this.controls.diskOuter.value = this.state.diskOuterRg.toString();
      this._updateTelemetry();
    });
    this.controls.doppler.addEventListener("input", (event) => {
      this.state.dopplerBoost = parseFloat(event.target.value);
      this._updateTelemetry();
    });
    this.controls.exposure.addEventListener("input", (event) => {
      this.state.exposure = parseFloat(event.target.value);
      this._updateTelemetry();
    });
    this.controls.flow.addEventListener("input", (event) => {
      this.state.flowRate = parseFloat(event.target.value);
      this._updateTelemetry();
    });
    this.controls.cameraAzimuth.addEventListener("input", (event) => {
      this.state.cameraAzimuthDeg = parseFloat(event.target.value);
      this._updateTelemetry();
    });
    this.controls.cameraElevation.addEventListener("input", (event) => {
      this.state.cameraElevationDeg = parseFloat(event.target.value);
      this._updateTelemetry();
    });
    this.controls.cameraZoom.addEventListener("input", (event) => {
      this.state.cameraZoom = parseFloat(event.target.value);
      this._updateTelemetry();
    });
  }

  _bindReset() {
    if (!this.resetButton) return;
    this.resetButton.addEventListener("click", () => this._resetState());
  }

  _bindTabs() {
    if (!this.tabButtons.length) return;
    this.tabButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const target = button.dataset.tabTarget;
        this._activateTab(target);
      });
    });
  }

  _activateTab(target) {
    this.tabButtons.forEach((btn) => {
      const isActive = btn.dataset.tabTarget === target;
      btn.classList.toggle("is-active", isActive);
      btn.setAttribute("aria-selected", String(isActive));
    });
    Object.entries(this.tabPanels).forEach(([key, panel]) => {
      if (!panel) return;
      panel.classList.toggle("is-active", key === target);
    });
  }

  _resetState() {
    this.state = { ...defaultState };
    this._syncInputsFromState();
    this._updateTelemetry();
  }

  _syncInputsFromState() {
    const assign = (control, value) => {
      if (control) control.value = value;
    };
    assign(this.controls.mass, this.state.massSolar / MASS_SLIDER_SCALE);
    assign(this.controls.spin, this.state.spin);
    assign(this.controls.inclination, this.state.inclinationDeg);
    assign(this.controls.distance, this.state.observerRg);
    assign(this.controls.diskInner, this.state.diskInnerRg);
    assign(this.controls.diskOuter, this.state.diskOuterRg);
    assign(this.controls.doppler, this.state.dopplerBoost);
    assign(this.controls.exposure, this.state.exposure);
    assign(this.controls.flow, this.state.flowRate);
    assign(this.controls.cameraZoom, this.state.cameraZoom);
    this._syncCameraControls();
    this._updateControlDisplays();
  }

  _syncCameraControls() {
    const setValue = (control, value, digits = 1) => {
      if (!control) return;
      control.value = Number(value).toFixed(digits);
    };
    setValue(this.controls.cameraAzimuth, wrap360(this.state.cameraAzimuthDeg), 1);
    setValue(this.controls.cameraElevation, wrap360(this.state.cameraElevationDeg), 1);
    setValue(this.controls.cameraZoom, this.state.cameraZoom, 2);
  }

  _updateControlDisplays() {
    const entries = [
      ["mass", this.state.massSolar / MASS_SLIDER_SCALE, 1, (val) => `${Number(val).toFixed(1)} M☉`],
      ["spin", this.state.spin, 3],
      ["inclination", this.state.inclinationDeg, 1],
      ["distance", this.state.observerRg, 0],
      ["diskInner", this.state.diskInnerRg, 2],
      ["diskOuter", this.state.diskOuterRg, 2],
      ["doppler", this.state.dopplerBoost, 2],
      ["exposure", this.state.exposure, 2],
      ["flow", this.state.flowRate, 2],
      ["camAzimuth", wrap360(this.state.cameraAzimuthDeg), 1],
      ["camElevation", wrap360(this.state.cameraElevationDeg), 1],
      ["camZoom", this.state.cameraZoom, 2],
    ];
    entries.forEach(([key, value, digits, formatter]) => {
      const target = this.controlValues[key];
      if (!target) return;
      if (typeof formatter === "function") {
        target.textContent = formatter(value);
      } else {
        target.textContent = Number(value).toFixed(digits);
      }
    });
  }

  _initInteractions() {
    const canvas = this.canvas;
    const interaction = {
      dragging: false,
      lastX: 0,
      lastY: 0,
    };

    const endDrag = () => {
      if (!interaction.dragging) return;
      interaction.dragging = false;
      canvas.classList.remove("is-dragging");
    };

    canvas.addEventListener("pointerdown", (event) => {
      interaction.dragging = true;
      interaction.lastX = event.clientX;
      interaction.lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add("is-dragging");
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!interaction.dragging) return;
      const deltaX = event.clientX - interaction.lastX;
      const deltaY = event.clientY - interaction.lastY;
      interaction.lastX = event.clientX;
      interaction.lastY = event.clientY;

      this.state.cameraAzimuthDeg = wrap360(
        this.state.cameraAzimuthDeg + deltaX * 0.3
      );
      this.state.cameraElevationDeg = wrap360(
        this.state.cameraElevationDeg + deltaY * 0.25
      );
      this._syncCameraControls();
      this._updateTelemetry();
    });

    const releasePointer = (event) => {
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      endDrag();
    };

    canvas.addEventListener("pointerup", releasePointer);
    canvas.addEventListener("pointerleave", releasePointer);
    canvas.addEventListener("pointercancel", releasePointer);

    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const delta = event.deltaY * -0.0015;
        this.state.cameraZoom = clamp(
          this.state.cameraZoom + delta,
          0.5,
          2.2
        );
        this._syncCameraControls();
        this._updateTelemetry();
      },
      { passive: false }
    );
  }

  _updateTelemetry() {
    const gravRadius = gravitationalRadiusMeters(this.state.massSolar);
    const isco = iscoRadiusRg(this.state.spin);
    const photon = photonRadiusRg(this.state.spin);
    const redshift = gravitationalRedshift(photon + 0.05, this.state.spin);
    const beta = orbitalBeta(isco, this.state.spin);

    const formatter = Intl.NumberFormat("en-US", {
      maximumFractionDigits: 3,
    });

    this.metrics.mass.textContent = `${formatter.format(
      this.state.massSolar
    )} M☉`;
    this.metrics.spin.textContent = formatter.format(this.state.spin);
    this.metrics.isco.textContent = `${formatter.format(isco)} Rg`;
    this.metrics.photon.textContent = `${formatter.format(photon)} Rg`;
    this.metrics.redshift.textContent = `${formatter.format(redshift)} @β=${formatter.format(
      beta
    )}`;

    this._updateControlDisplays();
    this._renderCalculations({
      massSolar: this.state.massSolar,
      spin: this.state.spin,
      gravRadius,
      isco,
      photon,
      redshift,
      beta,
      observer: this.state.observerRg,
    });

    this.renderer.updateState({
      spin: this.state.spin,
      inclination: toRadians(this.state.inclinationDeg),
      observerRg: this.state.observerRg,
      gravRadius,
      diskInnerRg: this.state.diskInnerRg,
      diskOuterRg: this.state.diskOuterRg,
      dopplerBoost: this.state.dopplerBoost,
      exposure: this.state.exposure,
      flowRate: this.state.flowRate,
      cameraAzimuth: toRadians(this.state.cameraAzimuthDeg),
      cameraElevation: toRadians(this.state.cameraElevationDeg),
      cameraZoom: this.state.cameraZoom,
    });
  }

  _renderCalculations(payload) {
    if (!this.calcReadout) return;
    const { massSolar, spin, gravRadius, isco, photon, redshift, beta, observer } = payload;
    const massKg = massSolar * SOLAR_MASS;
    const fmt = (value, digits = 3) =>
      Number(value).toExponential(digits);
    const fmtFixed = (value, digits = 3) =>
      Number(value).toFixed(digits);

    const a = clamp(spin, -0.998, 0.998);
    const term = Math.cbrt(1 - a * a);
    const z1 = 1 + term * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
    const z2 = Math.sqrt(3 * a * a + z1 * z1);
    const photonAngle = (2 / 3) * Math.acos(-Math.abs(a));
    const photonFormula = 2 * (1 + Math.cos(photonAngle));
    const redshiftR = photon + 0.05;

    const cards = [
      {
        title: "重力半徑",
        lines: [
          "R_g = 2GM / c²",
          `= 2 × ${G} × ${fmt(massKg)} / (${C}²)`,
          `= ${fmt(gravRadius)} m`,
        ],
      },
      {
        title: "ISCO (Bardeen)",
        lines: [
          "Z₁ = 1 + (1 - a²)^{1/3}[ (1+a)^{1/3} + (1-a)^{1/3} ]",
          ` = ${fmtFixed(z1, 4)}`,
          "Z₂ = √(3a² + Z₁²) = " + fmtFixed(z2, 4),
          "r_ISCO = 3 + Z₂ - sign(a)√[(3-Z₁)(3+Z₁+2Z₂)]",
          ` = ${fmtFixed(isco, 4)} R_g`,
        ],
      },
      {
        title: "光子球",
        lines: [
          "r_ph = 2(1 + cos(2/3·acos(-|a|)))",
          ` = ${fmtFixed(photonFormula, 4)} R_g`,
        ],
      },
      {
        title: "紅移",
        lines: [
          "z = 1/√(1 - 2/r + a²/r²) - 1",
          `r = r_ph + 0.05 = ${fmtFixed(redshiftR, 3)} R_g`,
          `z = ${fmtFixed(redshift, 4)}`,
        ],
      },
      {
        title: "軌道 β",
        lines: [
          `ω = 1 / (r^{3/2} + a)`,
          `β = r·ω / c`,
          ` = ${fmtFixed(beta, 4)}`,
        ],
      },
    ];

    this.calcReadout.innerHTML = cards
      .map(
        (card) => `
        <article class="calc-card">
          <h3>${card.title}</h3>
          <pre>${card.lines.join("\n")}</pre>
        </article>`
      )
      .join("");
    this.latestTelemetry = { massSolar, spin, gravRadius, isco, photon, redshift, beta, observer };
    this._updateFormulaPanel(this.latestTelemetry);
    if (this.consoleLive) {
      this._evaluateFormulas(false);
    }
  }

  _updateFormulaPanel(payload) {
    if (!this.formulaPanel || !payload) return;
    const { massSolar, gravRadius, spin, isco, photon, redshift, beta } = payload;
    const massKg = massSolar * SOLAR_MASS;
    this.formulaPanel.textContent = `R_g = 2GM / c²
  = 2 × ${G} × ${massKg.toExponential(3)} / ${C}²
  = ${gravRadius.toExponential(4)} m

r_ISCO(a=${spin.toFixed(3)}) = ${isco.toFixed(4)} R_g
r_ph = 2[1 + cos(2/3·acos(-|a|))] = ${photon.toFixed(4)} R_g

z = 1/√(1 - 2/r + a²/r²) - 1
  r = r_ph + 0.05 → z = ${redshift.toFixed(4)}

β = r·ω/c,  ω = 1/(r^{3/2} + a) → β = ${beta.toFixed(4)}`;
  }

  _initFormulaConsole() {
    if (!this.calcConsole || !this.formulaInput) return;
    this.consoleToggle?.addEventListener("click", () => {
      this.consoleLive = !this.consoleLive;
      this.consoleToggle.classList.toggle("is-active", this.consoleLive);
      this.consoleToggle.textContent = this.consoleLive ? "LIVE" : "PAUSE";
    });
    this.consoleFullscreen?.addEventListener("click", () => {
      const active = this.calcConsole.classList.toggle("fullscreen");
      this.consoleFullscreen.textContent = active ? "⤺" : "⤢";
      if (!active) {
        this.calcConsole.style.left = "";
        this.calcConsole.style.top = "";
        this.calcConsole.style.cursor = "";
      }
    });
    if (this.calcConsole) {
      let dragging = false;
      let startOffset = { x: 0, y: 0 };
      this.calcConsole.addEventListener("pointerdown", (event) => {
        if (!this.calcConsole.classList.contains("fullscreen")) return;
        const header = event.target.closest(".console-header");
        const actionArea = event.target.closest(".console-header-actions");
        if (!header || actionArea) return;
        dragging = true;
        const rect = this.calcConsole.getBoundingClientRect();
        startOffset = {
          x: event.clientX - rect.left,
          y: event.clientY - rect.top,
        };
        this.calcConsole.setPointerCapture(event.pointerId);
        this.calcConsole.style.cursor = "grabbing";
      });
      const endDrag = (event) => {
        if (!dragging) return;
        dragging = false;
        this.calcConsole.releasePointerCapture(event.pointerId);
        this.calcConsole.style.cursor = "grab";
      };
      this.calcConsole.addEventListener("pointermove", (event) => {
        if (!dragging) return;
        this.calcConsole.style.left = `${event.clientX - startOffset.x}px`;
        this.calcConsole.style.top = `${event.clientY - startOffset.y}px`;
      });
      this.calcConsole.addEventListener("pointerup", endDrag);
      this.calcConsole.addEventListener("pointerleave", endDrag);
    }
    this.consoleInfo?.addEventListener("click", () => {
      if (!this.consoleTooltip) return;
      this.consoleTooltip.classList.toggle("visible");
    });
    if (this.consoleTooltip) {
      let dragging = false;
      let moved = false;
      let offset = { x: 0, y: 0 };
      this.consoleTooltip.addEventListener("pointerdown", (event) => {
        dragging = true;
        moved = false;
        const rect = this.consoleTooltip.getBoundingClientRect();
        offset = {
          x: event.clientX - rect.left,
          y: event.clientY - rect.top,
        };
        this.consoleTooltip.setPointerCapture(event.pointerId);
        this.consoleTooltip.style.cursor = "grabbing";
      });
      const finish = (event) => {
        if (!dragging) return;
        dragging = false;
        this.consoleTooltip.releasePointerCapture(event.pointerId);
        this.consoleTooltip.style.cursor = "grab";
      };
      this.consoleTooltip.addEventListener("pointermove", (event) => {
        if (!dragging) return;
        moved = true;
        this.consoleTooltip.style.left = `${event.clientX - offset.x}px`;
        this.consoleTooltip.style.top = `${event.clientY - offset.y}px`;
      });
      this.consoleTooltip.addEventListener("pointerup", finish);
      this.consoleTooltip.addEventListener("pointerleave", finish);
      this.consoleTooltip.addEventListener("click", () => {
        if (!moved) {
          this.consoleTooltip.classList.remove("visible");
        }
      });
    }
    this.applyFormulaBtn?.addEventListener("click", () => this._evaluateFormulas(true));
    this.clearFormulaBtn?.addEventListener("click", () => {
      if (!this.formulaInput) return;
      this.formulaInput.value =
        "doppler = sqrt((1+beta)/(1-beta))\ninclination = 50\nobserver = 45";
      this._evaluateFormulas(false);
    });
    this.formulaInput.addEventListener("input", () => this._updateSuggestionList());
    this.formulaInput.addEventListener("keydown", (event) => {
      if (event.key === "Tab" && this.suggestionList?.classList.contains("visible")) {
        event.preventDefault();
        const first = this.suggestionList.querySelector(".suggestion-item");
        if (first) {
          this._insertSuggestion(first.dataset.insert);
        }
      }
    });
    this.formulaInput.addEventListener("blur", () => {
      window.setTimeout(() => this.suggestionList?.classList.remove("visible"), 120);
    });
  }

  _evaluateFormulas(applyUpdates = false) {
    if (!this.formulaInput || !this.formulaOutput || !this.latestTelemetry) return;
    const content = this.formulaInput.value.trim();
    if (!content) {
      this.formulaOutput.textContent = "輸入公式，如 doppler = sqrt((1+beta)/(1-beta))";
      return;
    }
    const { spin, gravRadius, isco, photon, redshift, beta } = this.latestTelemetry;
    const context = {
      r: photon,
      a: spin,
      beta,
      incl: toRadians(this.state.inclinationDeg),
      Rg: gravRadius,
      ISCO: isco,
      photon,
      z: redshift,
      observer: this.state.observerRg,
    };
    const assignments = {};
    const outputs = [];
    const lines = content.split(/\n+/);
    for (const row of lines) {
      const line = row.trim();
      if (!line) continue;
      const [labelRaw, exprRaw] = line.split("=").map((part) => part?.trim());
      if (!labelRaw || !exprRaw) {
        outputs.push(`無法解析：${line}`);
        continue;
      }
      try {
        const fn = new Function(
          "r",
          "a",
          "beta",
          "incl",
          "Rg",
          "ISCO",
          "photon",
          "z",
          "observer",
          `return ${exprRaw};`
        );
        const value = fn(
          context.r,
          context.a,
          context.beta,
          context.incl,
          context.Rg,
          context.ISCO,
          context.photon,
          context.z,
          context.observer
        );
        outputs.push(`${labelRaw} = ${Number(value).toFixed(6)}`);
        assignments[labelRaw.toLowerCase()] = Number(value);
      } catch (error) {
        outputs.push(`${labelRaw}: 錯誤 (${error.message})`);
      }
    }
    this.formulaOutput.textContent = outputs.join("\n");
    if (applyUpdates) {
      this._applyFormulaAssignments(assignments);
    }
  }

  _updateSuggestionList() {
    if (!this.formulaInput || !this.suggestionList) return;
    const caret = this.formulaInput.selectionStart;
    const text = this.formulaInput.value.slice(0, caret);
    const token = text.split(/[\s=()+*\/-]+/).pop();
    if (!token || token.length < 2) {
      this.suggestionList.classList.remove("visible");
      return;
    }
    const matcher = token.toLowerCase();
    const matches = this.formulaSuggestions
      .filter((item) => item.label.toLowerCase().includes(matcher))
      .slice(0, 5);
    if (!matches.length) {
      this.suggestionList.classList.remove("visible");
      return;
    }
    this.suggestionList.innerHTML = "";
    matches.forEach((item) => {
      const div = document.createElement("div");
      div.className = "suggestion-item";
      div.textContent = item.label;
      div.dataset.insert = item.insert;
      div.addEventListener("click", () => this._insertSuggestion(item.insert));
      this.suggestionList.appendChild(div);
    });
    this.suggestionList.classList.add("visible");
  }

  _insertSuggestion(insertText) {
    if (!this.formulaInput) return;
    const start = this.formulaInput.selectionStart;
    const end = this.formulaInput.selectionEnd;
    const before = this.formulaInput.value.slice(0, start);
    const after = this.formulaInput.value.slice(end);
    const token = before.split(/[\s=()+*\/-]+/).pop() || "";
    const newBefore = before.slice(0, before.length - token.length) + insertText;
    this.formulaInput.value = newBefore + after;
    const newCaret = newBefore.length;
    this.formulaInput.setSelectionRange(newCaret, newCaret);
    this._updateSuggestionList();
    this.suggestionList?.classList.remove("visible");
  }

  _applyFormulaAssignments(assignments) {
    const setIfDefined = (key, fn) => {
      if (assignments[key] === undefined || Number.isNaN(assignments[key])) return;
      fn(assignments[key]);
    };
    setIfDefined("mass", (val) => {
      this.state.massSolar = clamp(val, 1e6, 5e10);
      if (this.controls.mass) {
        this.controls.mass.value = (this.state.massSolar / MASS_SLIDER_SCALE).toFixed(0);
      }
    });
    setIfDefined("spin", (val) => {
      this.state.spin = clamp(val, 0, 0.998);
    });
    setIfDefined("inclination", (val) => {
      this.state.inclinationDeg = clamp(val, 0, 90);
    });
    setIfDefined("observer", (val) => {
      this.state.observerRg = clamp(val, 5, 400);
    });
    setIfDefined("distance", (val) => {
      this.state.observerRg = clamp(val, 5, 400);
    });
    setIfDefined("diskinner", (val) => {
      this.state.diskInnerRg = clamp(val, 0.5, 10);
    });
    setIfDefined("diskouter", (val) => {
      this.state.diskOuterRg = clamp(val, this.state.diskInnerRg + 0.2, 60);
    });
    setIfDefined("doppler", (val) => {
      this.state.dopplerBoost = clamp(val, 0.5, 3);
    });
    setIfDefined("exposure", (val) => {
      this.state.exposure = clamp(val, 0.6, 1.6);
    });
    setIfDefined("flow", (val) => {
      this.state.flowRate = clamp(val, 0.5, 3);
    });
    setIfDefined("zoom", (val) => {
      this.state.cameraZoom = clamp(val, 0.5, 2.2);
    });
    setIfDefined("camazimuth", (val) => {
      this.state.cameraAzimuthDeg = wrap360(val);
    });
    setIfDefined("camelevation", (val) => {
      this.state.cameraElevationDeg = wrap360(val);
    });
    this._syncInputsFromState();
    this._updateTelemetry();
  }

  _bindIntroOverlay() {
    if (!this.overlay || !this.initButton) return;
    this.initButton.addEventListener("click", () => {
      this.overlay.classList.add("is-hidden");
      this.overlay.setAttribute("aria-hidden", "true");
      this._ensureAudioContext();
      this._playNarration(
        "initial",
        "觀測鏈已啟動，載入 Kerr 幾何與語音敘述。調整控制台以更新吸積盤狀態。"
      );
    });
  }

  _createNarrationClips() {
    const make = (path) => {
      const clip = new Audio(path);
      clip.preload = "auto";
      clip.crossOrigin = "anonymous";
      clip.volume = 0.85;
      return clip;
    };
    return {
      initial: make("audio/Initial Narration.mp3"),
      intro: make("audio/narrate-intro.mp3"),
      lensing: make("audio/narrate-lensing.mp3"),
      horizon: make("audio/narrate-horizon.mp3"),
    };
  }

  _bindAudioControls() {
    this.ambienceToggle?.classList.toggle("is-active", this.isAmbienceOn);
    if (this.audioButtons.intro) {
      this.audioButtons.intro.addEventListener("click", () => {
        this._ensureAudioContext();
        this._playNarration(
          "intro",
          "黑洞是時空彎曲到連光也無法逃逸的區域，事件視界定義了無人返程的邊界。"
        );
      });
    }
    if (this.audioButtons.lensing) {
      this.audioButtons.lensing.addEventListener("click", () => {
        this._ensureAudioContext();
        this._playNarration(
          "lensing",
          "注意光暈與背面鏡像，這是重力透鏡讓盤面後方的光線在我們視線上彎折。"
        );
      });
    }
    if (this.audioButtons.horizon) {
      this.audioButtons.horizon.addEventListener("click", () => {
        this._ensureAudioContext();
        this._playNarration(
          "horizon",
          "吸積盤以相對論速度旋轉，靠近我們的一側受 Doppler 加亮並偏藍，遠離的一側則偏紅。"
        );
      });
    }
    if (this.ambienceToggle) {
      this.ambienceToggle.addEventListener("click", () => {
        this._ensureAudioContext();
        this._toggleAmbience();
      });
    }
  }

  _ensureAudioContext() {
    if (this.audioCtx) return;
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) return;
    this.audioCtx = new AudioContextCtor();
    this._startDrone();
  }

  _startDrone() {
    if (!this.audioCtx || this.droneGain) return;
    const osc1 = this.audioCtx.createOscillator();
    const osc2 = this.audioCtx.createOscillator();
    const gain = this.audioCtx.createGain();
    const filter = this.audioCtx.createBiquadFilter();

    osc1.type = "sine";
    osc1.frequency.value = 50;

    osc2.type = "sawtooth";
    osc2.frequency.value = 0.2;
    const modGain = this.audioCtx.createGain();
    modGain.gain.value = 20;
    osc2.connect(modGain);
    modGain.connect(osc1.frequency);

    filter.type = "lowpass";
    filter.frequency.value = 200;

    osc1.connect(filter);
    filter.connect(gain);
    gain.connect(this.audioCtx.destination);
    gain.gain.value = 0.3;

    osc1.start();
    osc2.start();
    this.droneGain = gain;
  }

  _toggleAmbience() {
    this.isAmbienceOn = !this.isAmbienceOn;
    if (this.droneGain && this.audioCtx) {
      this.droneGain.gain.setTargetAtTime(
        this.isAmbienceOn ? 0.3 : 0,
        this.audioCtx.currentTime,
        0.5
      );
    }
    this.ambienceToggle?.classList.toggle("is-active", this.isAmbienceOn);
  }

  _showSubtitle(text) {
    if (!this.subtitleText) return;
    this.subtitleText.textContent = text;
    this.subtitleText.style.opacity = "1";
    clearTimeout(this.subtitleTimer);
    this.subtitleTimer = window.setTimeout(() => {
      this.subtitleText.style.opacity = "0";
    }, 6000);
  }

  _playNarration(id, subtitle) {
    this._showSubtitle(subtitle);
    const clip = this.narrationClips?.[id];
    if (!clip) return;
    Object.values(this.narrationClips).forEach((audio) => {
      audio.pause();
      audio.currentTime = 0;
    });
    clip.play().catch(() => {});
  }

  attachSolverBridge(bridge) {
    this.solverBridge = bridge;
    this.statusPill.textContent = "WASM 求解器已注入";
    this.statusPill.style.borderColor = "rgba(67,198,247,0.6)";
  }

  reportValidation(resultText) {
    const target = document.getElementById("verification-summary");
    target.textContent = resultText;
  }

  async _pollValidation() {
    try {
      const response = await fetch("validation/report.json", {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error("尚未產生驗證報告");
      }
      const data = await response.json();
      const verdict = data.ok ? "PASS" : "FAIL";
      this.reportValidation(
        `${verdict} max_err=${data.max_rel_error.toExponential(2)}`
      );
    } catch (error) {
      this.reportValidation(error.message);
    }
  }

  _loop(timestamp) {
    const delta = timestamp - this.lastFrame;
    this.lastFrame = timestamp;
    this.frame += 1;
    this.accumTime += delta;
    if (this.accumTime >= 500) {
      const fps = (this.frame / this.accumTime) * 1000;
      this.metrics.fps.textContent = fps.toFixed(1);
      this.frame = 0;
      this.accumTime = 0;
    }

    if (this.solverBridge && typeof this.solverBridge.update === "function") {
      if (timestamp - this.lastSolverTick > 120) {
        this.solverBridge.update(this.state);
        this.lastSolverTick = timestamp;
      }
    }

    if (this.solverBridge?.hasPayload()) {
      const energy = this.solverBridge.consumeEnergyEstimate();
      this.renderer.pushSolverEnergy(energy);
    } else {
      const fallbackEnergy = 0.25 + 0.15 * Math.sin(timestamp * 0.001);
      this.renderer.pushSolverEnergy(fallbackEnergy);
    }

    this.renderer.render(timestamp * 0.001);
    requestAnimationFrame((t) => this._loop(t));
  }
}

function init() {
  try {
    const app = new BlackHoleCard();
    window.blackHoleCard = app;
  } catch (error) {
    const canvas = document.getElementById("bh-canvas");
    canvas.insertAdjacentHTML(
      "beforebegin",
      `<p class="error">初始化失敗：${error.message}</p>`
    );
    console.error(error);
  }
}

document.addEventListener("DOMContentLoaded", init);

class KerrWasmBridge {
  constructor(Module, options = {}) {
    this.Module = Module;
    this.samples = options.samples || 128;
    this.impactRange = options.impactRange || [-40, 40];
    this.energy = 0;
    this.initialized = false;
  }

  init() {
    if (!this.Module?._malloc) {
      throw new Error("WASM 模組缺少 malloc");
    }
    this.configPtr = this.Module._malloc(40);
    this.bufferSizeBytes = this.samples * 5 * Float64Array.BYTES_PER_ELEMENT;
    this.bufferPtr = this.Module._malloc(this.bufferSizeBytes);
    this.traceFn = this.Module.cwrap("trace_kerr_bundle", "number", [
      "number",
      "number",
      "number",
    ]);
    this.initialized = true;
  }

  update(state) {
    if (!this.initialized) {
      this.init();
    }
    const view = new DataView(this.Module.HEAPU8.buffer, this.configPtr, 40);
    view.setFloat64(0, this.impactRange[0], true);
    view.setFloat64(8, this.impactRange[1], true);
    view.setFloat64(16, state.spin, true);
    view.setFloat64(24, state.observerRg, true);
    view.setInt32(32, this.samples, true);

    const count = this.traceFn(this.configPtr, this.bufferPtr, this.samples);
    if (count <= 0) {
      this.energy = 0;
      return;
    }

    const sampleView = new Float64Array(
      this.Module.HEAPF64.buffer,
      this.bufferPtr,
      count * 5
    );
    let accum = 0;
    for (let i = 0; i < count; i += 5) {
      const deflection = sampleView[i + 1];
      const closest = sampleView[i + 3];
      const hit = sampleView[i + 4];
      accum += Math.exp(-closest * 0.2) * (1 + 0.5 * hit) * Math.abs(deflection);
    }
    this.energy = Math.min(accum / count, 3);
  }

  hasPayload() {
    return this.initialized && Number.isFinite(this.energy) && this.energy > 0;
  }

  consumeEnergyEstimate() {
    return this.energy;
  }
}

window.attachKerrWasmModule = async (moduleFactoryPromise) => {
  const app = window.blackHoleCard;
  try {
    const Module =
      typeof moduleFactoryPromise === "function"
        ? await moduleFactoryPromise()
        : await moduleFactoryPromise;
    const bridge = new KerrWasmBridge(Module, { samples: 96 });
    if (app) {
      app.attachSolverBridge(bridge);
    }
  } catch (error) {
    console.error("載入 WASM 失敗", error);
  }
};
