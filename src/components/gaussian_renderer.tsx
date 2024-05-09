import React from "react";
import { createShader, createProgram, createTexture, createRGBA32FTexture, createDepth32FTexture, } from '../utils/webgl';
import { Camera, TrackballCamera, } from '../utils/camera';
import { mat4, quat } from 'gl-matrix';
import { arrayMax, deg2rad } from "../utils/math";
import { GaussianScene, loadPly } from "./gaussian_renderer_utils/sceneLoader";
import Form from 'react-bootstrap/Form';
import { genPlane } from "../utils/uvplane";
import { Vertex } from "../utils/vertex";
import { polyline, polyline_facing_point } from "../utils/polyline";

type GaussianRendererProps = {
  style?: React.CSSProperties,
  className?: string
  width: number,
  height: number
}


const gsengine_vs = `#version 300 es
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

const gsengine_fs = `#version 300 es
precision highp float;

in vec3 col;
in float scale_modif;
in float depth;
in vec4 con_o;
in vec2 xy;
in vec2 pixf;

layout(location=0) out vec4 fragColor;
layout(location=1) out vec4 fragInvDepth;

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
    
    if (alpha < 1./255.) {
        discard;
    }

    // Eq. (3) from 3D Gaussian splatting paper.
    fragColor = vec4(col * alpha, alpha);
    fragInvDepth = vec4(vec3(1.0/depth) * alpha, alpha);
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

class GaussianRendererEngine {
  // canvas to render to
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;

  // worker
  private sortWorker: Worker;
  private sortWorkerBusy: boolean = false;
  private lastSortedViewProjMatrix: mat4 | null = null;

  // ply data to render
  private plyData: GaussianScene | null = null;
  private loadedSortedScene: GaussianScene | null = null;

  // uniforms
  private wLoc: WebGLUniformLocation;
  private hLoc: WebGLUniformLocation;
  private focalXLoc: WebGLUniformLocation;
  private focalYLoc: WebGLUniformLocation;
  private tanFovXLoc: WebGLUniformLocation;
  private tanFovYLoc: WebGLUniformLocation;
  private scaleModifierLoc: WebGLUniformLocation;
  private projMatrixLoc: WebGLUniformLocation;
  private viewMatrixLoc: WebGLUniformLocation;
  private boxMinLoc: WebGLUniformLocation;
  private boxMaxLoc: WebGLUniformLocation;

  // vertex array object
  private vao: WebGLVertexArrayObject;
  // buffers
  private buffers!: {
    color: WebGLBuffer,
    center: WebGLBuffer,
    opacity: WebGLBuffer,
    covA: WebGLBuffer,
    covB: WebGLBuffer,
  }

  private xsize: number;
  private ysize: number;

  public fbo: WebGLFramebuffer;
  public col_tex: WebGLTexture;
  public inv_depth_tex: WebGLTexture;

  public get_xsize = () => this.xsize;
  public get_ysize = () => this.ysize;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.program = createProgram(
      this.gl,
      [
        createShader(this.gl, this.gl.VERTEX_SHADER, gsengine_vs),
        createShader(this.gl, this.gl.FRAGMENT_SHADER, gsengine_fs),
      ]
    )!;
    this.gl.useProgram(this.program);

    const setupAttributeBuffer = (name: string, components: number) => {
      const location = this.gl.getAttribLocation(this.program, name)
      const buffer = this.gl.createBuffer()!
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer)
      this.gl.enableVertexAttribArray(location)
      this.gl.vertexAttribPointer(location, components, this.gl.FLOAT, false, 0, 0)
      this.gl.vertexAttribDivisor(location, 1)
      return buffer
    }

    this.vao = this.gl.createVertexArray()!;
    this.gl.bindVertexArray(this.vao);

    // Create attribute buffers
    this.buffers = {
      color: setupAttributeBuffer('a_col', 3),
      center: setupAttributeBuffer('a_center', 3),
      opacity: setupAttributeBuffer('a_opacity', 1),
      covA: setupAttributeBuffer('a_covA', 3),
      covB: setupAttributeBuffer('a_covB', 3),
    }

    this.wLoc = this.gl.getUniformLocation(this.program, 'W')!;
    this.hLoc = this.gl.getUniformLocation(this.program, 'H')!;
    this.focalXLoc = this.gl.getUniformLocation(this.program, 'focal_x')!;
    this.focalYLoc = this.gl.getUniformLocation(this.program, 'focal_y')!;
    this.tanFovXLoc = this.gl.getUniformLocation(this.program, 'tan_fovx')!;
    this.tanFovYLoc = this.gl.getUniformLocation(this.program, 'tan_fovy')!;
    this.scaleModifierLoc = this.gl.getUniformLocation(this.program, 'scale_modifier')!;
    this.projMatrixLoc = this.gl.getUniformLocation(this.program, 'projmatrix')!;
    this.viewMatrixLoc = this.gl.getUniformLocation(this.program, 'viewmatrix')!;
    this.boxMinLoc = this.gl.getUniformLocation(this.program, 'boxmin')!;
    this.boxMaxLoc = this.gl.getUniformLocation(this.program, 'boxmax')!;

    this.xsize = this.gl.canvas.width;
    this.ysize = this.gl.canvas.height;

    // create color texture
    this.fbo = this.gl.createFramebuffer()!;
    // this makes fbo the current active framebuffer
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo);

    // configure the currently active framebuffer to use color texture as color attachment 0 
    this.col_tex = createTexture(this.gl, this.xsize, this.ysize)!;
    this.gl.framebufferTexture2D(
      this.gl.FRAMEBUFFER, // will bind as a framebuffer
      this.gl.COLOR_ATTACHMENT0, // Attaches the texture to the framebuffer's color buffer. 
      this.gl.TEXTURE_2D, // we have a 2d texture
      this.col_tex, // the texture to attach
      0 // the mipmap level (we don't want mipmapping, so we set to 0)
    );
    this.inv_depth_tex = createRGBA32FTexture(this.gl, this.xsize, this.ysize, new Float32Array(this.xsize * this.ysize * 4))!;
    this.gl.framebufferTexture2D(
      this.gl.FRAMEBUFFER, // will bind as a framebuffer
      this.gl.COLOR_ATTACHMENT1, // Attaches the texture to the framebuffer's depth buffer. 
      this.gl.TEXTURE_2D, // we have a 2d texture
      this.inv_depth_tex, // the texture to attach
      0 // the mipmap level (we don't want mipmapping, so we set to 0)
    );

    // init sort worker
    this.sortWorker = new Worker(new URL('../components/gaussian_renderer_utils/sortWorker.ts', import.meta.url), { type: 'module' });
    this.sortWorker.onmessage = (e) => {
      this.sortWorkerBusy = false;
      this.recieveUpdatedGaussianData(e.data.data);
    }
  }

  setScene = (data: GaussianScene, camera: Camera) => {
    this.plyData = data;
    this.doWorkerSort(camera.viewProjMatrix(this.gl.canvas.width, this.gl.canvas.height));
  }

  doWorkerSort = (viewProjMatrix: mat4) => {
    this.sortWorkerBusy = true;
    this.lastSortedViewProjMatrix = viewProjMatrix;
    this.sortWorker.postMessage({
      viewMatrix: this.lastSortedViewProjMatrix,
      sortingAlgorithm: 'Array.sort',
      gaussians: this.plyData
    });
  }

  // recieve ordered gaussian data from worker
  recieveUpdatedGaussianData = (data: GaussianScene) => {
    const updateBuffer = (buffer: WebGLBuffer, data: Float32Array) => {
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer)
      this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.STATIC_DRAW)
    }
    updateBuffer(this.buffers.color, data.colors)
    updateBuffer(this.buffers.center, data.positions)
    updateBuffer(this.buffers.opacity, data.opacities)
    updateBuffer(this.buffers.covA, data.cov3Da)
    updateBuffer(this.buffers.covB, data.cov3Db)

    this.loadedSortedScene = data;
  }

  render = (camera: Camera) => {
    this.gl.useProgram(this.program);

    // settings
    this.gl.disable(this.gl.DEPTH_TEST)
    this.gl.enable(this.gl.BLEND);
    this.gl.blendFunc(this.gl.ONE_MINUS_DST_ALPHA, this.gl.ONE)

    // bind the framebuffer
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo);
    this.gl.viewport(0, 0, this.xsize, this.ysize);

    // bind the vao (vertex buffers)
    this.gl.bindVertexArray(this.vao);

    const W = this.xsize;
    const H = this.ysize;
    const tan_fovy = Math.tan(camera.fov() * 0.5)
    const tan_fovx = tan_fovy * W / H
    const focal_y = H / (2 * tan_fovy)
    const focal_x = W / (2 * tan_fovx)

    const viewMatrix = camera.viewMatrix();
    const viewProjMatrix = camera.viewProjMatrix(W, H)

    this.gl.uniform1f(this.wLoc, W);
    this.gl.uniform1f(this.hLoc, H);
    this.gl.uniform1f(this.focalXLoc, focal_x);
    this.gl.uniform1f(this.focalYLoc, focal_y);
    this.gl.uniform1f(this.tanFovXLoc, tan_fovx);
    this.gl.uniform1f(this.tanFovYLoc, tan_fovy);
    this.gl.uniform1f(this.scaleModifierLoc, 1.0);
    this.gl.uniformMatrix4fv(this.viewMatrixLoc, false, convertViewMatrixTargetCoordinateSystem(viewMatrix));
    this.gl.uniformMatrix4fv(this.projMatrixLoc, false, convertViewProjectionMatrixTargetCoordinateSystem(viewProjMatrix));

    if (this.loadedSortedScene) {
      this.gl.uniform3fv(this.boxMinLoc, this.loadedSortedScene.sceneMin)
      this.gl.uniform3fv(this.boxMaxLoc, this.loadedSortedScene.sceneMax)
      // draw triangles
      this.gl.clear(this.gl.DEPTH_BUFFER_BIT | this.gl.COLOR_BUFFER_BIT);
      this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0, this.gl.COLOR_ATTACHMENT1]);
      this.gl.drawArraysInstanced(this.gl.TRIANGLE_STRIP, 0, 4, this.loadedSortedScene.count);
    }
  }

  update = (camera: Camera) => {
    const viewProjMatrix = camera.viewProjMatrix(this.gl.canvas.width, this.gl.canvas.height);
    if (this.loadedSortedScene && this.lastSortedViewProjMatrix) {
      const f = mat4.frob(mat4.subtract(mat4.create(), viewProjMatrix, this.lastSortedViewProjMatrix));
      if (f > 0.1 && !this.sortWorkerBusy) {
        this.doWorkerSort(viewProjMatrix);
      }
    }
  }

  cleanup = () => {
    this.sortWorker.terminate();
    this.gl.deleteFramebuffer(this.fbo);
    this.gl.deleteTexture(this.col_tex);
    this.gl.deleteTexture(this.inv_depth_tex);
    this.gl.deleteProgram(this.program);
    this.gl.deleteBuffer(this.buffers.color);
    this.gl.deleteBuffer(this.buffers.center);
    this.gl.deleteBuffer(this.buffers.opacity);
    this.gl.deleteBuffer(this.buffers.covA);
    this.gl.deleteBuffer(this.buffers.covB);
  }
}



// renders triangles with a position and color attribute
const overlay_vs = `#version 300 es
layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_color;

uniform mat4 u_view;
uniform mat4 u_view_proj;

out vec3 v_color;
out float v_depth;

void main() {
  vec4 pos = u_view * vec4(a_position, 1);
  v_depth = pos.z/pos.w;
  v_color = a_color;
  gl_Position = u_view_proj * vec4(a_position, 1);
}`

const overlay_fs = `#version 300 es
precision highp float;

in vec3 v_color;
in float v_depth;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outInvDepth;

void main() {
  outColor = vec4(v_color, 1);
  outInvDepth = vec4(vec3(1.0/v_depth), 1);
}`;

type DrawableObject = {
  vertexes: Vertex[]
}

class OverlayEngine {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;

  private vao: WebGLVertexArrayObject;
  private positionBuffer: WebGLBuffer;
  private colorBuffer: WebGLBuffer;

  private viewLoc: WebGLUniformLocation;
  private viewProjLoc: WebGLUniformLocation;


  public fbo: WebGLFramebuffer;
  public col_tex: WebGLTexture;
  // this is the depth we want to use as input for compositing
  public inv_depth_tex: WebGLTexture;
  // this is used internally by webgl and has no blending (used for depth testing)
  private webgl_depth_tex: WebGLTexture;


  private xsize: number;
  private ysize: number;

  public get_xsize = () => this.xsize;
  public get_ysize = () => this.ysize;

  private n_vertexes: number = 0;
  private objects: DrawableObject[] = [];

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.program = createProgram(
      this.gl,
      [
        createShader(this.gl, this.gl.VERTEX_SHADER, overlay_vs),
        createShader(this.gl, this.gl.FRAGMENT_SHADER, overlay_fs),
      ]
    )!;
    this.gl.useProgram(this.program);

    const setupAttributeBuffer = (name: string, components: number) => {
      const location = this.gl.getAttribLocation(this.program, name)
      const buffer = this.gl.createBuffer()!
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer)
      this.gl.enableVertexAttribArray(location)
      this.gl.vertexAttribPointer(location, components, this.gl.FLOAT, false, 0, 0)
      return buffer
    }

    this.vao = this.gl.createVertexArray()!;
    this.gl.bindVertexArray(this.vao);

    this.positionBuffer = setupAttributeBuffer('a_position', 3);
    this.colorBuffer = setupAttributeBuffer('a_color', 3);

    this.viewLoc = this.gl.getUniformLocation(this.program, 'u_view')!;
    this.viewProjLoc = this.gl.getUniformLocation(this.program, 'u_view_proj')!;

    this.xsize = this.gl.canvas.width;
    this.ysize = this.gl.canvas.height;

    // create color texture
    this.fbo = this.gl.createFramebuffer()!;
    // this makes fbo the current active framebuffer
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo);

    // configure the currently active framebuffer to use color texture as color attachment 0 
    this.col_tex = createTexture(this.gl, this.xsize, this.ysize)!;
    this.gl.framebufferTexture2D(
      this.gl.FRAMEBUFFER, // will bind as a framebuffer
      this.gl.COLOR_ATTACHMENT0, // Attaches the texture to the framebuffer's color buffer. 
      this.gl.TEXTURE_2D, // we have a 2d texture
      this.col_tex, // the texture to attach
      0 // the mipmap level (we don't want mipmapping, so we set to 0)
    );
    this.inv_depth_tex = createRGBA32FTexture(this.gl, this.xsize, this.ysize, new Float32Array(this.xsize * this.ysize * 4))!;
    this.gl.framebufferTexture2D(
      this.gl.FRAMEBUFFER, // will bind as a framebuffer
      this.gl.COLOR_ATTACHMENT1, // Attaches the texture to the framebuffer's color buffer. 
      this.gl.TEXTURE_2D, // we have a 2d texture
      this.inv_depth_tex, // the texture to attach
      0 // the mipmap level (we don't want mipmapping, so we set to 0)
    );
    this.webgl_depth_tex = createDepth32FTexture(this.gl, this.xsize, this.ysize, new Float32Array(this.xsize * this.ysize))!;
    this.gl.framebufferTexture2D(
      this.gl.FRAMEBUFFER, // will bind as a framebuffer
      this.gl.DEPTH_ATTACHMENT, // Attaches the texture to the framebuffer's depth buffer. 
      this.gl.TEXTURE_2D, // we have a 2d texture
      this.webgl_depth_tex, // the texture to attach
      0 // the mipmap level (we don't want mipmapping, so we set to 0)
    );
  }

  // updates the position and color buffers
  reconstructBuffers = () => {
    this.n_vertexes = this.objects.reduce((acc, obj) => acc + obj.vertexes.length, 0);
    const positions = new Float32Array(this.n_vertexes * 3);
    const colors = new Float32Array(this.n_vertexes * 3);

    let offset = 0;
    for (const obj of this.objects) {
      for (const vertex of obj.vertexes) {
        positions.set(vertex.position, offset * 3);
        colors.set(vertex.color, offset * 3);
        offset += 1;
      }
    }

    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.colorBuffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, colors, this.gl.STATIC_DRAW);
  }


  addObject = (obj: DrawableObject) => {
    this.objects.push(obj);
    this.reconstructBuffers();
  }

  render = (camera: Camera) => {
    this.gl.disable(this.gl.BLEND);
    this.gl.enable(this.gl.DEPTH_TEST)

    this.gl.useProgram(this.program);

    this.gl.uniformMatrix4fv(this.viewLoc, false, convertViewMatrixTargetCoordinateSystem(camera.viewMatrix()));
    this.gl.uniformMatrix4fv(this.viewProjLoc, false, convertViewProjectionMatrixTargetCoordinateSystem(camera.viewProjMatrix(this.xsize, this.ysize)));

    this.gl.bindVertexArray(this.vao);

    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo);
    this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0, this.gl.COLOR_ATTACHMENT1]);

    this.gl.clear(this.gl.DEPTH_BUFFER_BIT | this.gl.COLOR_BUFFER_BIT);
    this.gl.viewport(0, 0, this.xsize, this.ysize);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, this.n_vertexes);
  }
}

const QUAD_BUFFER = new Float32Array(genPlane(1, 1).flatMap(v => [v[0], v[1]]));



const compositor_vs = `#version 300 es
in vec2 a_position;
out vec2 v_texCoord;

void main() {
  v_texCoord = a_position;

  // convert from 0->1 to 0->2
  // convert from 0->2 to -1->+1 (clip space)
  vec2 clipSpace = (a_position * 2.0) - 1.0;

  gl_Position = vec4(clipSpace, 0, 1);
}`

const compositor_fs = `#version 300 es
precision highp float;
precision highp sampler2D;

// texture 0
uniform sampler2D u_render_tex0;
uniform sampler2D u_inv_depth_tex0;
// texture 1
uniform sampler2D u_render_tex1;
uniform sampler2D u_inv_depth_tex1;

in vec2 v_texCoord;
out vec4 v_outColor;

void main() {
  float inv_depth0 = texture(u_inv_depth_tex0, v_texCoord).r;
  float inv_depth1 = texture(u_inv_depth_tex1, v_texCoord).r;

  if (inv_depth0 > inv_depth1) {
    v_outColor = texture(u_render_tex0, v_texCoord);
  } else {
    v_outColor = texture(u_render_tex1, v_texCoord);
  }
  // v_outColor = vec4(vec3(max(depth0-depth1, 0.0)), 1.0);
}`;


// Always draws to the canvas
class Compositor {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;

  private vao: WebGLVertexArrayObject;

  private positionBuffer: WebGLBuffer;

  private renderTex0Loc: WebGLUniformLocation;
  private invDepthTex0Loc: WebGLUniformLocation;
  private renderTex1Loc: WebGLUniformLocation;
  private invDepthTex1Loc: WebGLUniformLocation;

  private xsize: number;
  private ysize: number;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.program = createProgram(
      this.gl,
      [
        createShader(this.gl, this.gl.VERTEX_SHADER, compositor_vs),
        createShader(this.gl, this.gl.FRAGMENT_SHADER, compositor_fs),
      ]
    )!;
    this.gl.useProgram(this.program);

    const setupAttributeBuffer = (name: string, components: number) => {
      const location = this.gl.getAttribLocation(this.program, name)
      const buffer = this.gl.createBuffer()!
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer)
      this.gl.enableVertexAttribArray(location)
      this.gl.vertexAttribPointer(location, components, this.gl.FLOAT, false, 0, 0)
      return buffer
    }

    this.vao = this.gl.createVertexArray()!;
    this.gl.bindVertexArray(this.vao);

    this.positionBuffer = setupAttributeBuffer('a_position', 2);
    gl.bufferData(gl.ARRAY_BUFFER, QUAD_BUFFER, gl.STATIC_DRAW);

    this.renderTex0Loc = this.gl.getUniformLocation(this.program, 'u_render_tex0')!;
    this.invDepthTex0Loc = this.gl.getUniformLocation(this.program, 'u_inv_depth_tex0')!;
    this.renderTex1Loc = this.gl.getUniformLocation(this.program, 'u_render_tex1')!;
    this.invDepthTex1Loc = this.gl.getUniformLocation(this.program, 'u_inv_depth_tex1')!;

    this.xsize = this.gl.canvas.width;
    this.ysize = this.gl.canvas.height;
  }

  render = (renderTex0: WebGLTexture, invDepthTex0: WebGLTexture, renderTex1: WebGLTexture, invDepthTex1: WebGLTexture) => {
    this.gl.useProgram(this.program);

    this.gl.bindVertexArray(this.vao);

    this.gl.viewport(0, 0, this.xsize, this.ysize);

    this.gl.uniform1i(this.renderTex0Loc, 0);
    this.gl.uniform1i(this.invDepthTex0Loc, 1);
    this.gl.uniform1i(this.renderTex1Loc, 2);
    this.gl.uniform1i(this.invDepthTex1Loc, 3);

    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, renderTex0);
    this.gl.activeTexture(this.gl.TEXTURE1);
    this.gl.bindTexture(this.gl.TEXTURE_2D, invDepthTex0);
    this.gl.activeTexture(this.gl.TEXTURE2);
    this.gl.bindTexture(this.gl.TEXTURE_2D, renderTex1);
    this.gl.activeTexture(this.gl.TEXTURE3);
    this.gl.bindTexture(this.gl.TEXTURE_2D, invDepthTex1);

    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
  }

}


const viz_vs = `#version 300 es
in vec2 a_position;
out vec2 v_texCoord;

void main() {
  v_texCoord = a_position;

  // convert from 0->1 to 0->2
  // convert from 0->2 to -1->+1 (clip space)
  vec2 clipSpace = (a_position * 2.0) - 1.0;

  gl_Position = vec4(clipSpace, 0, 1);
}`

const viz_fs = `#version 300 es
precision highp float;
precision highp sampler2D;

// the rendered texture
uniform sampler2D u_render_tex;

in vec2 v_texCoord;
out vec4 v_outColor;

void main() {
  v_outColor = vec4(texture(u_render_tex, v_texCoord).rgb, 1.0);
}`;



type VizData = {
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  texLoc: WebGLUniformLocation,
  tex: WebGLTexture
}

// TODO: learn how to handle error cases

type GaussianEditorState = {}

class GaussianEditor extends React.Component<GaussianRendererProps, GaussianEditorState> {
  private canvas = React.createRef<HTMLCanvasElement>();
  private gl!: WebGL2RenderingContext;

  // visualization canvases
  private gsEngineColorViz = React.createRef<HTMLCanvasElement>();
  private gsEngineDepthViz = React.createRef<HTMLCanvasElement>();

  private overlayEngineColorViz = React.createRef<HTMLCanvasElement>();
  private overlayEngineDepthViz = React.createRef<HTMLCanvasElement>();

  // visualization canvas data
  private gsColorVizData!: VizData;
  private gsDepthVizData!: VizData;

  private overlayColorVizData!: VizData;
  private overlayDepthVizData!: VizData;

  private fileInput = React.createRef<HTMLInputElement>();

  private camera!: TrackballCamera;


  // renders gaussians
  private gsRendererEngine!: GaussianRendererEngine;
  // renders triangles
  private overlayEngine!: OverlayEngine;
  // composits to canvas
  private compositor!: Compositor;

  private requestID!: number;

  constructor(props: GaussianRendererProps) {
    super(props);
  }

  componentDidMount() {
    const canvas = this.canvas.current!;
    this.camera = new TrackballCamera(
      canvas,
      {
        rotation: quat.fromEuler(quat.create(), 0, 0.001, 0)
      }
    );
    this.gl = canvas.getContext('webgl2')!;
    this.gl.getExtension('EXT_color_buffer_float');
    this.gl.getExtension('EXT_float_blend');

    this.gsRendererEngine = new GaussianRendererEngine(this.gl);
    this.overlayEngine = new OverlayEngine(this.gl);
    this.compositor = new Compositor(this.gl);

    let circ_r2 = (theta: number): [number, number, number] => {
      return [2 * Math.cos(theta), 0, 2 * Math.sin(theta)];
    }

    this.overlayEngine.addObject({
      vertexes: polyline_facing_point(
        // points in the line
        [
          circ_r2(0 / 6 * 2 * Math.PI),
          circ_r2(1 / 6 * 2 * Math.PI),
          circ_r2(2 / 6 * 2 * Math.PI),
          circ_r2(3 / 6 * 2 * Math.PI),
          circ_r2(4 / 6 * 2 * Math.PI),
          circ_r2(5 / 6 * 2 * Math.PI),
        ],
        // width
        [
          0.1,
          0.1,
          0.1,
          0.1,
          0.1,
          0.1,
        ],
        // colors
        [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 0, 0],
          [0, 1, 0],
        ],
        [0, 0, 0]
      )
    });

    // set up viz canvases
    this.gsColorVizData = this.setupVizCanvas(this.gsEngineColorViz.current!);
    this.gsDepthVizData = this.setupVizCanvas(this.gsEngineDepthViz.current!);
    this.overlayColorVizData = this.setupVizCanvas(this.overlayEngineColorViz.current!);
    this.overlayDepthVizData = this.setupVizCanvas(this.overlayEngineDepthViz.current!);


    this.requestID = window.requestAnimationFrame(this.animationLoop);
  }

  setupVizCanvas = (canvas: HTMLCanvasElement): VizData => {
    const gl = canvas.getContext('webgl2')!;
    // setup a full canvas clip space quad
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, QUAD_BUFFER, gl.STATIC_DRAW);

    // setup viz program
    const viz_program = createProgram(
      gl,
      [
        createShader(gl, gl.VERTEX_SHADER, viz_vs),
        createShader(gl, gl.FRAGMENT_SHADER, viz_fs),
      ]
    )!;

    // set up position viz attributes
    const positionLoc = gl.getAttribLocation(viz_program, 'a_position');

    // setup our attributes to tell WebGL how to pull
    // the data from the buffer above to the position attribute
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(
      positionLoc,
      2,         // size (num components)
      this.gl.FLOAT,  // type of data in buffer
      false,     // normalize
      0,         // stride (0 = auto)
      0,         // offset
    );

    // texture uniform
    const texLoc = gl.getUniformLocation(viz_program, 'u_render_tex')!;

    // create texture to go along with it
    const tex = createTexture(gl, this.props.width, this.props.height)!;

    return {
      gl,
      program: viz_program,
      texLoc,
      tex
    }
  }


  visualizeTexture = (data: VizData, tex_data: Uint8Array) => {
    const gl = data.gl;
    gl.useProgram(data.program);

    // set the texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, data.tex);
    gl.uniform1i(data.texLoc, 0);

    // upload the texture data
    gl.texImage2D(
      gl.TEXTURE_2D,
      0, // mip level
      gl.RGBA, // internal format
      this.props.width,
      this.props.height,
      0, // border
      gl.RGBA, // format
      gl.UNSIGNED_BYTE, // type
      tex_data
    );

    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.clear(gl.DEPTH_BUFFER_BIT | gl.COLOR_BUFFER_BIT);
    // draw the quad
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  convertTex = (tex_data: Float32Array) => {
    const tex = new Uint8Array(tex_data.length);
    for (let i = 0; i < tex_data.length; i++) {
      tex[i] = Math.floor(tex_data[i] * 255);
    }
    return tex;
  }

  componentWillUnmount() {
    // stop animation loop
    window.cancelAnimationFrame(this.requestID!);
    // TODO: destroy vao, buffer, programs, shaders, etc
    this.camera.cleanup();
  }

  handleFileInputChange = async () => {
    const ply_file = this.fileInput.current?.files?.[0];
    if (ply_file) {
      this.gsRendererEngine.setScene(
        loadPly(await ply_file.arrayBuffer()),
        this.camera
      );
    }
  }


  animationLoop = () => {
    this.camera.update();
    this.gsRendererEngine.update(this.camera);

    // render gaussians to texture
    this.gsRendererEngine.render(this.camera);

    // copy color texture
    const color_tex_data = new Uint8Array(this.gsRendererEngine.get_xsize() * this.gsRendererEngine.get_ysize() * 4);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.gsRendererEngine.fbo);
    this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
    this.gl.readPixels(0, 0, this.gsRendererEngine.get_xsize(), this.gsRendererEngine.get_ysize(), this.gl.RGBA, this.gl.UNSIGNED_BYTE, color_tex_data);

    // copy depth texture
    const depth_tex_data = new Float32Array(this.gsRendererEngine.get_xsize() * this.gsRendererEngine.get_ysize() * 4);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.gsRendererEngine.fbo);
    this.gl.readBuffer(this.gl.COLOR_ATTACHMENT1);
    this.gl.readPixels(0, 0, this.gsRendererEngine.get_xsize(), this.gsRendererEngine.get_ysize(), this.gl.RGBA, this.gl.FLOAT, depth_tex_data);

    // visualize textures
    this.visualizeTexture(this.gsColorVizData, color_tex_data);
    this.visualizeTexture(this.gsDepthVizData, this.convertTex(depth_tex_data));

    // render overlay to texture
    this.overlayEngine.render(this.camera);

    // copy color texture
    const overlay_color_tex_data = new Uint8Array(this.overlayEngine.get_xsize() * this.overlayEngine.get_ysize() * 4);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.overlayEngine.fbo);
    this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
    this.gl.readPixels(0, 0, this.overlayEngine.get_xsize(), this.overlayEngine.get_ysize(), this.gl.RGBA, this.gl.UNSIGNED_BYTE, overlay_color_tex_data);

    // copy depth texture
    const overlay_depth_tex_data = new Float32Array(this.overlayEngine.get_xsize() * this.overlayEngine.get_ysize() * 4);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.overlayEngine.fbo);
    this.gl.readBuffer(this.gl.COLOR_ATTACHMENT1);
    this.gl.readPixels(0, 0, this.overlayEngine.get_xsize(), this.overlayEngine.get_ysize(), this.gl.RGBA, this.gl.FLOAT, overlay_depth_tex_data);

    // visualize textures
    this.visualizeTexture(this.overlayColorVizData, overlay_color_tex_data);
    this.visualizeTexture(this.overlayDepthVizData, this.convertTex(overlay_depth_tex_data));


    // composite
    this.compositor.render(
      this.gsRendererEngine.col_tex,
      this.gsRendererEngine.inv_depth_tex,
      this.overlayEngine.col_tex,
      this.overlayEngine.inv_depth_tex
    );

    this.requestID = window.requestAnimationFrame(this.animationLoop);
  }

  render() {
    return <>
      < canvas
        style={this.props.style}
        className={this.props.className}
        ref={this.canvas}
        height={this.props.height}
        width={this.props.width}
      />
      <canvas
        style={this.props.style}
        className={this.props.className}
        ref={this.gsEngineColorViz}
        height={this.props.height}
        width={this.props.width}
      />
      <canvas
        style={this.props.style}
        className={this.props.className}
        ref={this.gsEngineDepthViz}
        height={this.props.height}
        width={this.props.width}
      />
      <canvas
        style={this.props.style}
        className={this.props.className}
        ref={this.overlayEngineColorViz}
        height={this.props.height}
        width={this.props.width}
      />
      <canvas
        style={this.props.style}
        className={this.props.className}
        ref={this.overlayEngineDepthViz}
        height={this.props.height}
        width={this.props.width}
      />
      <Form.Group controlId="formFile" className="mb-3">
        <Form.Label>Select PLY File</Form.Label>
        <Form.Control ref={this.fileInput} type="file" accept=".ply" onChange={this.handleFileInputChange} />
      </Form.Group>
    </>
  }
}

export default GaussianEditor;