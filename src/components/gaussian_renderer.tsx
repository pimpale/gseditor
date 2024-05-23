import React from "react";
import { createShader, createProgram, createTexture, createRGBA32FTexture, createDepth32FTexture, createR32UITexture, } from '../utils/webgl';
import { Camera, TrackballCamera, } from '../utils/camera';
import { mat3, mat4, quat, vec2, vec3 } from 'gl-matrix';
import { arrayMax, deg2rad } from "../utils/math";
import { GaussianObjectInput, loadPly } from "./gaussian_renderer_utils/sceneLoader";
import Form from 'react-bootstrap/Form';
import { genPlane } from "../utils/uvplane";
import { Vertex } from "../utils/vertex";
import { polyline, polyline_facing_point } from "../utils/polyline";
import { ProcessedGaussianScene } from "./gaussian_renderer_utils/sortWorker";
import assert from "../utils/assert";
import { CanvasMouseTracker } from "../utils/canvas";
import { axesLines } from "../utils/axes";

const QUAD_BUFFER = new Float32Array(genPlane(1, 1).flatMap(v => [v[0], v[1]]));


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
layout(location=5) in uint a_objId;

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

out vec3 splat_color;
out float depth;
out float scale_modif;
out vec4 con_o;
out vec2 xy;
out vec2 pixf;
flat out uint objId;

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
    splat_color = a_col;
    con_o = vec4(conic, a_opacity);
    xy = point_image;
    pixf = screen_pos;
    depth = p_view.z;
    objId = a_objId;

    // (Webgl-specific) Convert from screen-space to clip-space
    vec2 clip_pos = screen_pos / vec2(W, H) * 2. - 1.;

    gl_Position = vec4(clip_pos, 0, 1);
}`

const gsengine_fs = `#version 300 es
precision highp float;

// time
uniform float time;

// which object is selected
uniform uint selectedObjectId;

// 0 = render all objects
// 1 = render only selected object
// 2 = highlight selected object
uniform uint selectedObjectRenderMode;

in vec3 splat_color;
in float scale_modif;
in float depth;
in vec4 con_o;
in vec2 xy;
in vec2 pixf;
flat in uint objId;

layout(location=0) out vec4 fragColor;
layout(location=1) out vec4 fragInvDepth;

vec3 depth_palette(float x) { 
    x = min(1., x);
    return vec3( sin(x*6.28/4.), x*x, mix(sin(x*6.28),x,.6) );
}

// Original CUDA implementation: https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L263
void main() {

    // discard splats with wrong objectId
    if (selectedObjectRenderMode == 1u && objId != selectedObjectId) {
        discard;
    }

    // highlight selected object
    vec3 color = splat_color;
    if (selectedObjectRenderMode == 2u && objId == selectedObjectId) {
      color = mix(splat_color, vec3(1.0, 0.0, 0.0), 0.1*mod(time + gl_FragCoord.x/10.0 + gl_FragCoord.y/10.0, 1.0));
    }

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

    float col_alpha = alpha;

    // with depth we want to avoid including the depth value of faint splats
    float depth_alpha = alpha*float(alpha > 50./255.);

    // Eq. (3) from 3D Gaussian splatting paper.
    fragColor = vec4(color * col_alpha, col_alpha);
    fragInvDepth = vec4(vec3(1.0/depth) * depth_alpha, depth_alpha);
}`;


const invertRowMat4 = (mat: mat4, row: number) => {
  mat[row + 0] = -mat[row + 0]
  mat[row + 4] = -mat[row + 4]
  mat[row + 8] = -mat[row + 8]
  mat[row + 12] = -mat[row + 12]
}

// converts a standard mat4 view matrix to the cursed coordinate system of gaussian splatting
const convertViewMatrixTargetCoordinateSystem = (vm: Readonly<mat4>) => {
  // copy the view matrix
  const viewMatrix = mat4.clone(vm)

  // invertRowMat4(viewMatrix, 0) // NOTE: inverting the x axis is webgl specific
  // invertRowMat4(viewMatrix, 1)
  invertRowMat4(viewMatrix, 2)

  return viewMatrix;
}

const convertViewProjectionMatrixTargetCoordinateSystem = (vpm: Readonly<mat4>) => {
  // copy the viewProjMatrix
  const viewProjMatrix = mat4.clone(vpm)

  // invertRowMat4(viewProjMatrix, 0) // NOTE: inverting the x axis is webgl specific
  // invertRowMat4(viewProjMatrix, 1)
  // invertRowMat4(viewProjMatrix, 2)

  return viewProjMatrix;
}


type GaussianRendererEngineSceneObject = {
  translation: vec3,
  rotation: quat,
  object: GaussianObjectInput,
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
  private sceneGraph: Map<number, GaussianRendererEngineSceneObject> = new Map();
  private processed_scenegraph: ProcessedGaussianScene | null = null;

  // whether we need to rebuild the processed scenegraph
  private needs_rebuild: boolean = true;

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
  private selectedObjectRenderModeLoc: WebGLUniformLocation;
  private selectedObjectIdLoc: WebGLUniformLocation;
  private timeLoc: WebGLUniformLocation;

  // vertex array object
  private vao: WebGLVertexArrayObject;

  // buffers
  private buffers: {
    color: WebGLBuffer,
    center: WebGLBuffer,
    opacity: WebGLBuffer,
    covA: WebGLBuffer,
    covB: WebGLBuffer,
    objId: WebGLBuffer,
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

    const setupAttributeBuffer = (name: string, ptr_type: number, components: number) => {
      const location = this.gl.getAttribLocation(this.program, name)
      const buffer = this.gl.createBuffer()
      assert(buffer !== null, "Failed to create buffer")
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer)
      this.gl.enableVertexAttribArray(location)
      if (ptr_type === this.gl.UNSIGNED_INT) {
        this.gl.vertexAttribIPointer(location, components, ptr_type, 0, 0)
      } else if (ptr_type === this.gl.FLOAT) {
        this.gl.vertexAttribPointer(location, components, ptr_type, false, 0, 0)
      } else {
        throw Error("ptr_type not supported")
      }
      this.gl.vertexAttribDivisor(location, 1)
      return buffer
    }

    this.vao = this.gl.createVertexArray()!;
    this.gl.bindVertexArray(this.vao);

    // Create attribute buffers
    this.buffers = {
      color: setupAttributeBuffer('a_col', this.gl.FLOAT, 3),
      center: setupAttributeBuffer('a_center', this.gl.FLOAT, 3),
      opacity: setupAttributeBuffer('a_opacity', this.gl.FLOAT, 1),
      covA: setupAttributeBuffer('a_covA', this.gl.FLOAT, 3),
      covB: setupAttributeBuffer('a_covB', this.gl.FLOAT, 3),
      objId: setupAttributeBuffer('a_objId', this.gl.UNSIGNED_INT, 1),
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
    this.timeLoc = this.gl.getUniformLocation(this.program, 'time')!;
    this.selectedObjectRenderModeLoc = this.gl.getUniformLocation(this.program, 'selectedObjectRenderMode')!;
    this.selectedObjectIdLoc = this.gl.getUniformLocation(this.program, 'selectedObjectId')!;

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

    // init sort worker
    this.sortWorker = new Worker(new URL('../components/gaussian_renderer_utils/sortWorker.ts', import.meta.url), { type: 'module' });
    this.sortWorker.onmessage = (e) => {
      this.sortWorkerBusy = false;
      this.recieveUpdatedGaussianData(e.data.data);
    }
  }

  getObject = (id: number) => {
    return this.sceneGraph.get(id);
  }

  addObject = (id: number, translate: vec3, rotate: quat, object: GaussianObjectInput) => {
    this.sceneGraph.set(id, { translation: translate, rotation: rotate, object });
    this.needs_rebuild = true;
  }

  setPositionObject = (id: number, translate: vec3) => {
    const obj = this.sceneGraph.get(id);
    if (obj === undefined) {
      throw Error(`Object with id ${id} does not exist`)
    }
    obj.translation = translate;
    this.needs_rebuild = true;
  }

  setRotationObject = (id: number, rotate: quat) => {
    const obj = this.sceneGraph.get(id);
    if (obj === undefined) {
      throw Error(`Object with id ${id} does not exist`)
    }
    obj.rotation = rotate;
    this.needs_rebuild = true;
  }

  removeObject = (id: number) => {
    this.sceneGraph.delete(id);
    this.needs_rebuild = true;
  }

  doWorkerSort = (viewProjMatrix: mat4) => {
    this.sortWorkerBusy = true;
    this.lastSortedViewProjMatrix = viewProjMatrix;

    this.sortWorker.postMessage({
      viewMatrix: this.lastSortedViewProjMatrix,
      sortingAlgorithm: 'count sort',
      sceneGraph: this.sceneGraph,
    });
  }

  // recieve ordered gaussian data from worker
  recieveUpdatedGaussianData = (data: ProcessedGaussianScene) => {
    const updateBuffer = (buffer: WebGLBuffer, data: Float32Array | Uint32Array) => {
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer)
      this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.STATIC_DRAW)
    }
    updateBuffer(this.buffers.color, data.colors)
    updateBuffer(this.buffers.center, data.positions)
    updateBuffer(this.buffers.opacity, data.opacities)
    updateBuffer(this.buffers.covA, data.cov3Da)
    updateBuffer(this.buffers.covB, data.cov3Db)
    updateBuffer(this.buffers.objId, data.objectIds)

    this.processed_scenegraph = data;
  }

  // returns a 3D array of the depths of the gaussian splats per object
  renderDepths = (camera: Camera): Map<number, Float32Array> => {
    const depth_buffers: Map<number, Float32Array> = new Map();

    // bind the framebuffer
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo);
    this.gl.viewport(0, 0, this.xsize, this.ysize);


    // draw gaussians
    this.gl.useProgram(this.program);

    // gaussian settings
    this.gl.disable(this.gl.DEPTH_TEST)
    this.gl.enable(this.gl.BLEND);
    this.gl.blendFunc(this.gl.ONE_MINUS_DST_ALPHA, this.gl.ONE)

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
    this.gl.uniform1ui(this.selectedObjectRenderModeLoc, 1);
    this.gl.uniform1f(this.timeLoc, performance.now() / 1000.0);

    for (const id of this.sceneGraph.keys()) {
      const depth_buffer = new Float32Array(W * H * 4);
      this.gl.uniform1ui(this.selectedObjectIdLoc, id);
      if (this.processed_scenegraph) {
        this.gl.uniform3fv(this.boxMinLoc, this.processed_scenegraph.sceneMin)
        this.gl.uniform3fv(this.boxMaxLoc, this.processed_scenegraph.sceneMax)

        // clear the framebuffer
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);

        // draw triangles
        this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0, this.gl.COLOR_ATTACHMENT1]);
        this.gl.drawArraysInstanced(this.gl.TRIANGLE_STRIP, 0, 4, this.processed_scenegraph.count);

        // read pixels from the framebuffer
        this.gl.readBuffer(this.gl.COLOR_ATTACHMENT1);
        this.gl.readPixels(0, 0, W, H, this.gl.RGBA, this.gl.FLOAT, depth_buffer);
      }
      depth_buffers.set(id, depth_buffer);
    }
    return depth_buffers;
  }


  render = (camera: Camera, selected_object_id: number | null) => {
    // bind the framebuffer
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo);
    this.gl.viewport(0, 0, this.xsize, this.ysize);

    // clear the framebuffer
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);

    // draw gaussians
    this.gl.useProgram(this.program);

    // gaussian settings
    this.gl.disable(this.gl.DEPTH_TEST)
    this.gl.enable(this.gl.BLEND);
    this.gl.blendFunc(this.gl.ONE_MINUS_DST_ALPHA, this.gl.ONE)

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

    if (selected_object_id !== null) {
      this.gl.uniform1ui(this.selectedObjectIdLoc, selected_object_id);
      this.gl.uniform1ui(this.selectedObjectRenderModeLoc, 2);
    } else {
      this.gl.uniform1ui(this.selectedObjectIdLoc, 0);
      this.gl.uniform1ui(this.selectedObjectRenderModeLoc, 0);
    }
    this.gl.uniform1f(this.wLoc, W);
    this.gl.uniform1f(this.hLoc, H);
    this.gl.uniform1f(this.focalXLoc, focal_x);
    this.gl.uniform1f(this.focalYLoc, focal_y);
    this.gl.uniform1f(this.tanFovXLoc, tan_fovx);
    this.gl.uniform1f(this.tanFovYLoc, tan_fovy);
    this.gl.uniform1f(this.scaleModifierLoc, 1.0);
    this.gl.uniformMatrix4fv(this.viewMatrixLoc, false, convertViewMatrixTargetCoordinateSystem(viewMatrix));
    this.gl.uniformMatrix4fv(this.projMatrixLoc, false, convertViewProjectionMatrixTargetCoordinateSystem(viewProjMatrix));
    this.gl.uniform1f(this.timeLoc, performance.now() / 1000.0);

    if (this.processed_scenegraph) {
      this.gl.uniform3fv(this.boxMinLoc, this.processed_scenegraph.sceneMin)
      this.gl.uniform3fv(this.boxMaxLoc, this.processed_scenegraph.sceneMax)
      // draw triangles
      this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0, this.gl.COLOR_ATTACHMENT1]);
      this.gl.drawArraysInstanced(this.gl.TRIANGLE_STRIP, 0, 4, this.processed_scenegraph.count);
    }
  }

  update = (camera: Camera) => {
    let needs_rebuild = this.needs_rebuild;
    const viewProjMatrix = camera.viewProjMatrix(this.gl.canvas.width, this.gl.canvas.height);
    if (this.processed_scenegraph && this.lastSortedViewProjMatrix) {
      const f = mat4.frob(mat4.subtract(mat4.create(), viewProjMatrix, this.lastSortedViewProjMatrix));
      if (f > 0.1) {
        needs_rebuild = true;
      }
    }

    if (needs_rebuild && !this.sortWorkerBusy) {
      this.doWorkerSort(viewProjMatrix);
      this.needs_rebuild = false;
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
    this.gl.deleteBuffer(this.buffers.objId);
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
  kind: "line" | "triangle",
  translation: vec3,
  rotation: quat,
  vertexes: Vertex[]
}

class OverlayEngine {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;

  private tri_vao: WebGLVertexArrayObject;
  private tri_n_vertexes: number; // number of vertexes in the buffers (may differ from the scene graph)
  private tri_positionBuffer: WebGLBuffer;
  private tri_colorBuffer: WebGLBuffer;

  private line_vao: WebGLVertexArrayObject;
  private line_n_vertexes: number; // number of vertexes in the buffers (may differ from the scene graph)
  private line_positionBuffer: WebGLBuffer;
  private line_colorBuffer: WebGLBuffer;

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

  private objects: Map<number, DrawableObject> = new Map();

  private needs_rebuild: boolean = true;

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

    this.tri_vao = this.gl.createVertexArray()!;
    this.gl.bindVertexArray(this.tri_vao);

    this.tri_n_vertexes = 0;
    this.tri_positionBuffer = setupAttributeBuffer('a_position', 3);
    this.tri_colorBuffer = setupAttributeBuffer('a_color', 3);


    this.line_vao = this.gl.createVertexArray()!;
    this.gl.bindVertexArray(this.line_vao);

    this.line_n_vertexes = 0;
    this.line_positionBuffer = setupAttributeBuffer('a_position', 3);
    this.line_colorBuffer = setupAttributeBuffer('a_color', 3);

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


  buildBuffers = (drawables: DrawableObject[]): [Float32Array, Float32Array, number] => {
    const n_vertexes = drawables.reduce((acc, obj) => acc + obj.vertexes.length, 0);
    const positions = new Float32Array(n_vertexes * 3);
    const colors = new Float32Array(n_vertexes * 3);

    let offset = 0;
    for (const obj of drawables) {
      for (const vertex of obj.vertexes) {
        positions.set(vertex.position, offset * 3);
        colors.set(vertex.color, offset * 3);
        offset += 1;
      }
    }

    return [positions, colors, n_vertexes]
  }


  // updates the position and color buffers
  update = () => {
    if (this.needs_rebuild) {
      const tri_drawables = [];
      const line_drawables = [];
      for (const obj of this.objects.values()) {
        if (obj.kind === "triangle") {
          tri_drawables.push(obj);
        } else if (obj.kind === "line") {
          line_drawables.push(obj);
        }
      }
      const [tri_positions, tri_colors, tri_n_vertexes] = this.buildBuffers(tri_drawables);
      const [line_positions, line_colors, line_n_vertexes] = this.buildBuffers(line_drawables);

      this.gl.bindVertexArray(this.tri_vao);
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.tri_positionBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, tri_positions, this.gl.STATIC_DRAW);
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.tri_colorBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, tri_colors, this.gl.STATIC_DRAW);
      this.tri_n_vertexes = tri_n_vertexes;

      this.gl.bindVertexArray(this.line_vao);
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.line_positionBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, line_positions, this.gl.STATIC_DRAW);
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.line_colorBuffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, line_colors, this.gl.STATIC_DRAW);
      this.line_n_vertexes = line_n_vertexes;

      this.needs_rebuild = false;
    }
  }


  addObject = (id: number, obj: DrawableObject) => {
    this.objects.set(id, obj);
    this.needs_rebuild = true;
  }

  setPositionObject = (id: number, translate: vec3) => {
    const obj = this.objects.get(id);
    if (obj === undefined) {
      throw Error(`Object with id ${id} does not exist`)
    }
    obj.translation = translate;
    this.needs_rebuild = true;
  }

  setRotationObject = (id: number, rotate: quat) => {
    const obj = this.objects.get(id);
    if (obj === undefined) {
      throw Error(`Object with id ${id} does not exist`)
    }
    obj.rotation = rotate;
    this.needs_rebuild = true;
  }

  removeObject = (id: number) => {
    this.objects.delete(id);
    this.needs_rebuild = true;
  }

  render = (camera: Camera) => {
    this.gl.disable(this.gl.BLEND);
    this.gl.enable(this.gl.DEPTH_TEST)

    this.gl.useProgram(this.program);

    this.gl.uniformMatrix4fv(this.viewLoc, false, convertViewMatrixTargetCoordinateSystem(camera.viewMatrix()));
    this.gl.uniformMatrix4fv(this.viewProjLoc, false, convertViewProjectionMatrixTargetCoordinateSystem(camera.viewProjMatrix(this.xsize, this.ysize)));

    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo);
    this.gl.drawBuffers([this.gl.COLOR_ATTACHMENT0, this.gl.COLOR_ATTACHMENT1]);

    this.gl.clear(this.gl.DEPTH_BUFFER_BIT | this.gl.COLOR_BUFFER_BIT);
    this.gl.viewport(0, 0, this.xsize, this.ysize);

    this.gl.bindVertexArray(this.tri_vao);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, this.tri_n_vertexes);


    if(Math.random() > 0.99) {
      for(const obj of this.objects.values()) {
        // print out the view matrix transformed vertices
        for(const vertex of obj.vertexes) {
          const pos = vec3.transformMat4(vec3.create(), vertex.position, camera.viewMatrix());
          console.log("1", pos);
          const pos2 = vec3.transformMat4(vec3.create(), pos, camera.viewProjMatrix(this.xsize, this.ysize));
          console.log("2", pos2);
        }
      }
    }

    this.gl.bindVertexArray(this.line_vao);
    this.gl.drawArrays(this.gl.LINES, 0, this.line_n_vertexes);

  }
}


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

function axisNameToIdx(axis: "x" | "y" | "z"): number {
  switch (axis) {
    case "x":
      return 0;
    case "y":
      return 1;
    case "z":
      return 2;
    default:
      throw Error("Invalid axis")
  }
}

function axisNameToVec3(axis: "x" | "y" | "z"): vec3 {
  switch (axis) {
    case "x":
      return vec3.fromValues(1, 0, 0);
    case "y":
      return vec3.fromValues(0, 1, 0);
    case "z":
      return vec3.fromValues(0, 0, 1);
    default:
      throw Error("Invalid axis")
  }
}


type IdleSelectedObjectState = {
  kind: "idle"
}

type TranslateSelectedObjectState = {
  kind: "translate",
}

type RotateSelectedObjectState = {
  kind: "rotate",
}

type TranslateWithAxisSelectedObjectState = {
  kind: "translate_with_axis",
  axis: "x" | "y" | "z",
  mouse_start: vec2,
  pos_start: vec3
}

type RotateWithAxisSelectedObjectState = {
  kind: "rotate_with_axis",
  axis: "x" | "y" | "z",
  mouse_start: vec2,
  quat_start: quat
}

type SelectedObjectInterfaceState = {
  kind: "selected_object_interface",
  selected_object_id: number,
  last_mouse_pos: vec2,
  selected_object_state: RotateSelectedObjectState | RotateWithAxisSelectedObjectState | TranslateSelectedObjectState | TranslateWithAxisSelectedObjectState | IdleSelectedObjectState;
}

type IdleInterfaceState = {
  kind: "idle"
}

type InterfaceState = IdleInterfaceState | SelectedObjectInterfaceState;

type MouseMoveEvent = {
  kind: "mousemove",
  location: vec2
}

type MouseClickEvent = {
  kind: "mouseclick",
  location: vec2
}

type KeyDownEvent = {
  kind: "keydown",
  key: string
}

type InterfaceInput = MouseMoveEvent | MouseClickEvent | KeyDownEvent;

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
  private cmt!: CanvasMouseTracker;

  private interface_state: InterfaceState = { kind: "idle" };
  private interface_inputs: InterfaceInput[] = [];

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

    this.overlayEngine.addObject(0, {
      kind: "line",
      translation: vec3.create(),
      rotation: quat.create(),
      vertexes: axesLines(4)
    })

    let circ_r2 = (theta: number): [number, number, number] => {
      return [2 * Math.cos(theta), 0, 2 * Math.sin(theta)];
    }
    this.overlayEngine.addObject(1, {
      kind: "triangle",
      translation: vec3.create(),
      rotation: quat.create(),
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

    this.cmt = new CanvasMouseTracker(canvas);
    this.cmt.addMouseClickListener(e => this.interface_inputs.push({ kind: "mouseclick", location: e }));
    this.cmt.addMouseMoveListener(e => this.interface_inputs.push({ kind: "mousemove", location: e }));
    this.cmt.addKeyDownListener(s => this.interface_inputs.push({ kind: "keydown", key: s }));

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

    gl.clear(gl.DEPTH_BUFFER_BIT | gl.COLOR_BUFFER_BIT);
    // draw the quad
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  convertRGBA32Tex = (tex_data: Float32Array) => {
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
      this.gsRendererEngine.addObject(
        Math.floor(Math.random() * 0xFFFFFFFF),
        vec3.fromValues(0, 0, 0),
        quat.create(),
        loadPly(await ply_file.arrayBuffer()),
      );
    }
  }

  getNearestObjectAt = (location: vec2): number | null => {
    const click_loc_idx = Math.floor(location[0]) + Math.floor(this.gsRendererEngine.get_ysize() - location[1]) * this.gsRendererEngine.get_xsize();
    const depth_arrs = this.gsRendererEngine.renderDepths(this.camera);
    let nearest_inv_depth = 0;
    let nearest_depth_obj: number | null = null;
    for (const [id, depth_arr] of depth_arrs) {
      const candidate_inv_depth = depth_arr[4 * click_loc_idx];
      if (candidate_inv_depth > nearest_inv_depth) {
        nearest_inv_depth = candidate_inv_depth;
        nearest_depth_obj = id;
      }
    }
    return nearest_depth_obj;
  }

  animationLoop = () => {
    this.camera.update();
    this.gsRendererEngine.update(this.camera);
    this.overlayEngine.update();

    // process inputs
    for (const input of this.interface_inputs) {
      console.log(this.interface_state)
      switch (input.kind) {
        case "mouseclick": {
          switch (this.interface_state.kind) {
            case "idle": {
              const nearest_depth_obj = this.getNearestObjectAt(input.location);
              if (nearest_depth_obj !== null) {
                this.interface_state = {
                  kind: "selected_object_interface",
                  selected_object_id: nearest_depth_obj,
                  selected_object_state: { kind: "idle" },
                  last_mouse_pos: input.location
                }
              }
              break;
            }
            case "selected_object_interface": {
              const nearest_depth_obj = this.getNearestObjectAt(input.location);
              if (nearest_depth_obj === this.interface_state.selected_object_id) {
                // this.interface_state = { kind: "idle" };
              } else if (nearest_depth_obj !== null) {
                this.interface_state = {
                  kind: "selected_object_interface",
                  selected_object_id: nearest_depth_obj,
                  selected_object_state: { kind: "idle" },
                  last_mouse_pos: input.location
                }
              }
              break;
            }
          }
          break;
        }
        case "keydown": {
          switch (this.interface_state.kind) {
            case "selected_object_interface": {
              switch (input.key) {
                case "r": {
                  this.interface_state.selected_object_state = { kind: "rotate", };
                  this.overlayEngine.removeObject(this.interface_state.selected_object_id);
                  this.overlayEngine.addObject(this.interface_state.selected_object_id, {
                    kind: "line",
                    translation: this.gsRendererEngine.getObject(this.interface_state.selected_object_id)!.translation,
                    rotation: quat.create(),
                    vertexes: axesLines(4)
                  })
                  break;
                }
                case "g": {
                  this.interface_state.selected_object_state = { kind: "translate", };
                  break;
                }
                case "x":
                case "y":
                case "z": {
                  const axis = input.key as "x" | "y" | "z";
                  if (this.interface_state.selected_object_state.kind === "rotate") {
                    this.interface_state.selected_object_state = {
                      kind: "rotate_with_axis",
                      axis,
                      quat_start: this.gsRendererEngine.getObject(this.interface_state.selected_object_id)!.rotation,
                      mouse_start: this.interface_state.last_mouse_pos
                    };
                  } else if (this.interface_state.selected_object_state.kind === "translate") {
                    this.interface_state.selected_object_state = {
                      kind: "translate_with_axis",
                      axis,
                      pos_start: this.gsRendererEngine.getObject(this.interface_state.selected_object_id)!.translation,
                      mouse_start: this.interface_state.last_mouse_pos
                    };
                  }
                  break;
                }
                case 'Escape': {
                  switch (this.interface_state.selected_object_state.kind) {
                    case "rotate_with_axis":
                    case "translate_with_axis":
                    case "rotate":
                    case "translate": {
                      this.interface_state.selected_object_state = { kind: "idle" };
                      this.overlayEngine.removeObject(this.interface_state.selected_object_id);
                      break;
                    }
                    case "idle": {
                      this.interface_state = { kind: "idle" };
                      break;
                    }
                  }
                  break;
                }
                default: {
                  console.log("unhandled key", input.key)
                  break;
                }
              }
              break;
            }
            default: {
              break;
            }
          }
          break;
        }
        case "mousemove": {
          switch (this.interface_state.kind) {
            case "selected_object_interface": {
              this.interface_state.last_mouse_pos = input.location;
              switch (this.interface_state.selected_object_state.kind) {
                case "rotate_with_axis": {
                  if (this.interface_state.selected_object_state.axis !== null) {
                    const mouse_delta = vec2.sub(vec2.create(), input.location, this.interface_state.selected_object_state.mouse_start);
                    const rotation = quat.setAxisAngle(quat.create(), axisNameToVec3(this.interface_state.selected_object_state.axis), mouse_delta[0] * 0.02);
                    this.gsRendererEngine.setRotationObject(this.interface_state.selected_object_id, quat.mul(quat.create(), this.interface_state.selected_object_state.quat_start, rotation));
                  }
                  break;
                }
                case "translate_with_axis": {
                  if (this.interface_state.selected_object_state.axis !== null) {
                    const mouse_delta = vec2.sub(vec2.create(), input.location, this.interface_state.selected_object_state.mouse_start);
                    const translation = vec3.clone(this.interface_state.selected_object_state.pos_start);
                    translation[axisNameToIdx(this.interface_state.selected_object_state.axis)] += mouse_delta[0] * 0.01;
                    this.gsRendererEngine.setPositionObject(this.interface_state.selected_object_id, translation);
                  }
                  break;
                }
                default: {
                  break;
                }
              }
              break;
            }
            default: {
              break;
            }
          }
          break;
        }
      }
    }
    this.interface_inputs = [];


    // render gaussians to texture
    const selected_object_id = this.interface_state.kind === "selected_object_interface"
      ? this.interface_state.selected_object_id
      : null;
    this.gsRendererEngine.render(this.camera, selected_object_id);

    // copy color texture
    const color_tex_data = new Uint8Array(this.gsRendererEngine.get_xsize() * this.gsRendererEngine.get_ysize() * 4);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.gsRendererEngine.fbo);
    this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0);
    this.gl.readPixels(0, 0, this.gsRendererEngine.get_xsize(), this.gsRendererEngine.get_ysize(), this.gl.RGBA, this.gl.UNSIGNED_BYTE, color_tex_data);

    //// copy depth texture
    const depth_tex_data = new Float32Array(this.gsRendererEngine.get_xsize() * this.gsRendererEngine.get_ysize() * 4);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.gsRendererEngine.fbo);
    this.gl.readBuffer(this.gl.COLOR_ATTACHMENT1);
    this.gl.readPixels(0, 0, this.gsRendererEngine.get_xsize(), this.gsRendererEngine.get_ysize(), this.gl.RGBA, this.gl.FLOAT, depth_tex_data);

    // visualize textures
    this.visualizeTexture(this.gsColorVizData, color_tex_data);
    this.visualizeTexture(this.gsDepthVizData, this.convertRGBA32Tex(depth_tex_data));

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
    this.visualizeTexture(this.overlayDepthVizData, this.convertRGBA32Tex(overlay_depth_tex_data));

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
        tabIndex={0}
        style={this.props.style}
        className={this.props.className}
        ref={this.canvas}
        height={this.props.height}
        width={this.props.width}
      />
      <br />
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