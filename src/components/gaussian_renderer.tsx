import React from "react";
import { createShader, createProgram } from '../utils/webgl';
import { TrackballCamera, } from '../utils/camera';
import { mat4, quat } from 'gl-matrix';
import { deg2rad } from "../utils/math";
import { LoadedPly } from "./gaussian_renderer_utils/sceneLoader";

type GaussianRendererProps = {
  style?: React.CSSProperties,
  className?: string
  width: number,
  height: number
}


const vs = `#version 300 es
layout(location=0) in vec3 a_center;
layout(location=1) in vec3 a_col;
layout(location=2) in float a_opacity;
layout(location=3) in vec3 a_covA;
layout(location=4) in vec3 a_covB;

uniform float W;
uniform float H;
uniform float focal_x;
uniform float focal_y;
uniform float tan_fovx;
uniform float tan_fovy;
uniform float scale_modifier;
uniform mat4 projmatrix;
uniform mat4 viewmatrix;
uniform vec3 boxmin;
uniform vec3 boxmax;

out vec3 col;
out float depth;
out float scale_modif;
out vec4 con_o;
out vec2 xy;
out vec2 pixf;

vec3 computeCov2D(vec3 mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, float[6] cov3D, mat4 viewmatrix) {
    vec4 t = viewmatrix * vec4(mean, 1.0);

    float limx = 1.3 * tan_fovx;
    float limy = 1.3 * tan_fovy;
    float txtz = t.x / t.z;
    float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    mat3 J = mat3(
        focal_x / t.z, 0, -(focal_x * t.x) / (t.z * t.z),
        0, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0
    );

    mat3 W =  mat3(
        viewmatrix[0][0], viewmatrix[1][0], viewmatrix[2][0],
        viewmatrix[0][1], viewmatrix[1][1], viewmatrix[2][1],
        viewmatrix[0][2], viewmatrix[1][2], viewmatrix[2][2]
    );

    mat3 T = W * J;

    mat3 Vrk = mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]
    );

    mat3 cov = transpose(T) * transpose(Vrk) * T;

    cov[0][0] += .3;
    cov[1][1] += .3;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

float ndc2Pix(float v, float S) {
    return ((v + 1.) * S - 1.) * .5;
}

#define hash33(p) fract(sin( (p) * mat3( 127.1,311.7,74.7 , 269.5,183.3,246.1 , 113.5,271.9,124.6) ) *43758.5453123)

// Original CUDA implementation: https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L156
void main() {
    vec3 p_orig = a_center;

    // Discard splats outside of the scene bounding box (should not happen)
    // if (p_orig.x < boxmin.x || p_orig.y < boxmin.y || p_orig.z < boxmin.z ||
    //     p_orig.x > boxmax.x || p_orig.y > boxmax.y || p_orig.z > boxmax.z) {
    //         gl_Position = vec4(0, 0, 0, 1);
    //         return;
    //     }

    // Transform point by projecting
    vec4 p_hom = projmatrix * vec4(p_orig, 1);
    float p_w = 1. / (p_hom.w + 1e-7);
    vec3 p_proj = p_hom.xyz * p_w;

    // Perform near culling, quit if outside.
    vec4 p_view = viewmatrix * vec4(p_orig, 1);
    if (p_view.z <= .4) {
        gl_Position = vec4(0, 0, 0, 1);
        return;
    }

    // (Webgl-specific) The covariance matrix is pre-computed on the CPU for faster performance
    float cov3D[6] = float[6](a_covA.x, a_covA.y, a_covA.z, a_covB.x, a_covB.y, a_covB.z);
    // computeCov3D(a_scale, scale_modifier, a_rot, cov3D);

    // Compute 2D screen-space covariance matrix
    vec3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

    // Invert covariance (EWA algorithm)
    float det = (cov.x * cov.z - cov.y * cov.y);
    if (det == 0.) {
        gl_Position = vec4(0, 0, 0, 1);
        return;
    }
    float det_inv = 1. / det;
    vec3 conic = vec3(cov.z, -cov.y, cov.x) * det_inv;

    // Compute extent in screen space (by finding eigenvalues of
    // 2D covariance matrix). Use extent to compute the bounding
    // rectangle of the splat in screen space.

    float mid = 0.5 * (cov.x + cov.z);
    float lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    float my_radius = ceil(3. * sqrt(max(lambda1, lambda2)));
    vec2 point_image = vec2(ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H));

    // (Webgl-specific) As the covariance matrix is calculated as a one-time operation on CPU in this implementation,
    // we need to apply the scale modifier differently to still allow for real-time scaling of the splats.
    my_radius *= .15 + scale_modifier * .85;
    scale_modif = 1. / scale_modifier;

    // (Webgl-specific) Convert gl_VertexID from [0,1,2,3] to [-1,-1],[1,-1],[-1,1],[1,1]
    vec2 corner = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) - 1.;
    // Vertex position in screen space
    vec2 screen_pos = point_image + my_radius * corner;

    // Store some useful helper data for the fragment stage
    col = a_col;
    con_o = vec4(conic, a_opacity);
    xy = point_image;
    pixf = screen_pos;
    depth = p_view.z;

    // (Webgl-specific) Convert from screen-space to clip-space
    vec2 clip_pos = screen_pos / vec2(W, H) * 2. - 1.;

    gl_Position = vec4(clip_pos, 0, 1);
}`

const fs = `#version 300 es
precision mediump float;

uniform bool show_depth_map;

in vec3 col;
in float scale_modif;
in float depth;
in vec4 con_o;
in vec2 xy;
in vec2 pixf;

out vec4 fragColor;

vec3 depth_palette(float x) { 
    x = min(1., x);
    return vec3( sin(x*6.28/4.), x*x, mix(sin(x*6.28),x,.6) );
}

// Original CUDA implementation: https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L263
void main() {

    // Resample using conic matrix (cf. "Surface 
    // Splatting" by Zwicker et al., 2001)
    vec2 d = xy - pixf;
    float power = -0.5 * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

    if (power > 0.) {
        discard;
    }

    // (Custom) As the covariance matrix is calculated in a one-time operation on CPU in this implementation,
    // we need to apply the scale modifier differently to still allow for real-time scaling of the splats.
    power *= scale_modif;

    // Eq. (2) from 3D Gaussian splatting paper.
    float alpha = min(.99f, con_o.w * exp(power));
    
    // (Custom) Colorize with depth value instead of color (z-buffer visualization)
    vec3 color = col;
    if (show_depth_map) {
        color = depth_palette(depth * .08);
    }

    if (alpha < 1./255.) {
        discard;
    }

    // Eq. (3) from 3D Gaussian splatting paper.
    fragColor = vec4(color * alpha, alpha);
}`;


const invertRow = (mat: mat4, row: number) => {
  mat[row + 0] = -mat[row + 0]
  mat[row + 4] = -mat[row + 4]
  mat[row + 8] = -mat[row + 8]
  mat[row + 12] = -mat[row + 12]
}

// converts a standard mat4 view matrix to the cursed coordinate system of gaussian splatting
const convertViewMatrixTargetCoordinateSystem = (vm: Readonly<mat4>) => {
  // copy the view matrix
  const viewMatrix = mat4.clone(vm)

  invertRow(viewMatrix, 0) // NOTE: inverting the x axis is webgl specific
  invertRow(viewMatrix, 1)
  invertRow(viewMatrix, 2)

  return viewMatrix;
}

const convertViewProjectionMatrixTargetCoordinateSystem = (vpm: Readonly<mat4>) => {
  // copy the viewProjMatrix
  const viewProjMatrix = mat4.clone(vpm)

  invertRow(viewProjMatrix, 0) // NOTE: inverting the x axis is webgl specific
  invertRow(viewProjMatrix, 1)

  return viewProjMatrix;
}

const FOV_Y = 60;

// TODO: learn how to handle error cases

type GaussianRendererState = {}

class GaussianRenderer extends React.Component<GaussianRendererProps, GaussianRendererState> {
  private canvas = React.createRef<HTMLCanvasElement>();
  private gl!: WebGL2RenderingContext;

  private camera!: TrackballCamera;

  // uniforms
  private wLoc!: WebGLUniformLocation;
  private hLoc!: WebGLUniformLocation;
  private focalXLoc!: WebGLUniformLocation;
  private focalYLoc!: WebGLUniformLocation;
  private tanFovXLoc!: WebGLUniformLocation;
  private tanFovYLoc!: WebGLUniformLocation;
  private scaleModifierLoc!: WebGLUniformLocation;
  private projMatrixLoc!: WebGLUniformLocation;
  private viewMatrixLoc!: WebGLUniformLocation;
  private boxMinLoc!: WebGLUniformLocation;
  private boxMaxLoc!: WebGLUniformLocation;


  // buffers
  private buffers!: {
    color: WebGLBuffer,
    center: WebGLBuffer,
    opacity: WebGLBuffer,
    covA: WebGLBuffer,
    covB: WebGLBuffer,
  }

  // ply data to render
  private loadedPlyData: LoadedPly | null = null;


  private requestID!: number;

  constructor(props: GaussianRendererProps) {
    super(props);
  }

  componentDidMount() {

    // init camera
    this.camera = new TrackballCamera(
      this.canvas.current!,
      {
        rotation: quat.fromEuler(quat.create(), 0, 0.1, 0)
      }
    );

    // get webgl
    this.gl = this.canvas.current!.getContext('webgl2', { premultipliedAlpha: false })!;

    const program = createProgram(
      this.gl,
      [
        createShader(this.gl, this.gl.VERTEX_SHADER, vs),
        createShader(this.gl, this.gl.FRAGMENT_SHADER, fs),
      ]
    )!;

    const setupAttributeBuffer = (name: string, components: number) => {
      const location = this.gl.getAttribLocation(program, name)
      const buffer = this.gl.createBuffer()!
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer)
      this.gl.enableVertexAttribArray(location)
      this.gl.vertexAttribPointer(location, components, this.gl.FLOAT, false, 0, 0)
      this.gl.vertexAttribDivisor(location, 1)
      return buffer
    }

    // Create attribute buffers
    this.buffers = {
      color: setupAttributeBuffer('a_col', 3),
      center: setupAttributeBuffer('a_center', 3),
      opacity: setupAttributeBuffer('a_opacity', 1),
      covA: setupAttributeBuffer('a_covA', 3),
      covB: setupAttributeBuffer('a_covB', 3),
    }

    // uniform float W;
    // uniform float H;
    // uniform float focal_x;
    // uniform float focal_y;
    // uniform float tan_fovx;
    // uniform float tan_fovy;
    // uniform float scale_modifier;
    // uniform mat4 projmatrix;
    // uniform mat4 viewmatrix;
    // uniform vec3 boxmin;
    // uniform vec3 boxmax;

    this.wLoc = this.gl.getUniformLocation(program, 'W')!;
    this.hLoc = this.gl.getUniformLocation(program, 'H')!;
    this.focalXLoc = this.gl.getUniformLocation(program, 'focal_x')!;
    this.focalYLoc = this.gl.getUniformLocation(program, 'focal_y')!;
    this.tanFovXLoc = this.gl.getUniformLocation(program, 'tan_fovx')!;
    this.tanFovYLoc = this.gl.getUniformLocation(program, 'tan_fovy')!;
    this.scaleModifierLoc = this.gl.getUniformLocation(program, 'scale_modifier')!;
    this.projMatrixLoc = this.gl.getUniformLocation(program, 'projmatrix')!;
    this.viewMatrixLoc = this.gl.getUniformLocation(program, 'viewmatrix')!;
    this.boxMinLoc = this.gl.getUniformLocation(program, 'boxmin')!;
    this.boxMaxLoc = this.gl.getUniformLocation(program, 'boxmax')!;

    this.gl.useProgram(program);

    // settings
    this.gl.disable(this.gl.DEPTH_TEST)
    this.gl.enable(this.gl.BLEND);
    this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

    // start animation loop
    this.animationLoop();
  }

  componentWillUnmount() {
    // stop animation loop
    window.cancelAnimationFrame(this.requestID!);
    // TODO: destroy vao, buffer, programs, shaders, etc
    this.camera.cleanup();
  }

  setPlyData = (plyData: LoadedPly) => {
    this.loadedPlyData = plyData;
    const updateBuffer = (buffer: WebGLBuffer, data: Float32Array) => {
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer)
      this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.DYNAMIC_DRAW)
    }

    updateBuffer(this.buffers.color, Float32Array.from(plyData.colors))
    updateBuffer(this.buffers.center, Float32Array.from(plyData.positions))
    updateBuffer(this.buffers.opacity, Float32Array.from(plyData.opacities))
    updateBuffer(this.buffers.covA, Float32Array.from(plyData.cov3Da))
    updateBuffer(this.buffers.covB, Float32Array.from(plyData.cov3Db))
  }

  animationLoop = () => {
    this.requestID = window.requestAnimationFrame(this.animationLoop);

    // set uniform
    const W = this.gl.canvas.width;
    const H = this.gl.canvas.height;
    const tan_fovy = Math.tan(FOV_Y * 0.5)
    const tan_fovx = tan_fovy * W / H
    const focal_y = H / (2 * tan_fovy)
    const focal_x = W / (2 * tan_fovx)

    const viewMatrix = this.camera.viewMatrix();

    const projMatrix = mat4.perspective(
      mat4.create(),
      deg2rad(FOV_Y),
      W / H,
      0.1,
      1000
    );

    this.gl.uniform1f(this.wLoc, W);
    this.gl.uniform1f(this.hLoc, H);
    this.gl.uniform1f(this.focalXLoc, focal_x);
    this.gl.uniform1f(this.focalYLoc, focal_y);
    this.gl.uniform1f(this.tanFovXLoc, tan_fovx);
    this.gl.uniform1f(this.tanFovYLoc, tan_fovy);
    this.gl.uniform1f(this.scaleModifierLoc, 1.0);
    this.gl.uniformMatrix4fv(this.viewMatrixLoc, false, convertViewMatrixTargetCoordinateSystem(viewMatrix));
    this.gl.uniformMatrix4fv(this.projMatrixLoc, false, convertViewProjectionMatrixTargetCoordinateSystem(mat4.multiply(mat4.create(), projMatrix, viewMatrix)));

    if (this.loadedPlyData) {
      this.gl.uniform3fv(this.boxMinLoc, this.loadedPlyData.sceneMin)
      this.gl.uniform3fv(this.boxMaxLoc, this.loadedPlyData.sceneMax)

      const n_gaussians = this.loadedPlyData.opacities.length;

      // draw triangles
      this.gl.drawArraysInstanced(this.gl.TRIANGLE_STRIP, 0, 4, n_gaussians);
    }



    this.camera.update();
  }

  render() {
    return <canvas
      style={this.props.style}
      className={this.props.className}
      ref={this.canvas}
      height={this.props.height}
      width={this.props.width}
    />
  }
}

export default GaussianRenderer;