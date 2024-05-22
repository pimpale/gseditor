import { glMatrix, quat, vec3, mat4 } from 'gl-matrix';
import { deg2rad } from './math';

export interface Camera {
  fov: () => number;
  viewMatrix: () => mat4;
  viewProjMatrix: (width: number, height: number) => mat4;
  update: () => void;
  cleanup: () => void;
} 



// https://www.xarg.org/2021/07/trackball-rotation-using-quaternions/
function projectTrackball(v: { x: number, y: number }) {
  const x = v.x;
  const y = v.y;

  const r = 1;

  let z;
  if (x * x + y * y <= r * r / 2) {
    z = Math.sqrt(r * r - (x * x) - (y * y));
  } else {
    z = (r * r / 2) / Math.sqrt(x * x + y * y);
  }

  return new Float32Array([x, -y, z]);
}

export type TrackballCameraOptions = {
  rotation?: quat,
  dampingFactor?: number,
}

type Point = {
  x: number,
  y: number
}

const FOV_Y = 47;

export class TrackballCamera implements Camera {
  // this is the rotation that would applied to the object every frame not being dragged
  private rotationQ;
  // how much to damp the rotation at each step
  private dampingFactor: number;

  private canvas: HTMLElement;

  // if mouse is pressed, start location of drag
  private mouseLoc: null | { start: Point, current: Point, previous: Point } = null;

  // the following two quaternions are multiplied together to produce the
  // real rotation

  // the base orientation of the object
  private baseQ = quat.create();
  // the rotation added by the mouse (generated each frame)
  private currQ = quat.create();
  // the rotation used for momentum
  private momentumQ = quat.create();

  // the offset radius to use
  private offsetRadius = 1;

  // when not being dragged, how much to show the momentum
  private currentMomentumLevel = 0;

  // normalizes the mouse coords such that the edge of the trackball is +-1
  getNormalizedMouseCoords = (e: { clientX: number, clientY: number }) => {
    const rect = this.canvas.getBoundingClientRect();
    // get client canvas x and y
    const client_cx = (rect.left + rect.right) / 2;
    const client_cy = (rect.top + rect.bottom) / 2;

    // get canvas width and height
    const width = rect.right - rect.left;
    const height = rect.bottom - rect.top;

    // the radius of our trackball will be the smaller of these 2
    const trackballRadius = Math.min(width, height);

    // get normalized mouse x and y
    const q = {
      x: 2 * (e.clientX - client_cx) / trackballRadius,
      y: 2 * (e.clientY - client_cy) / trackballRadius
    }
    return q;
  }

  fov = () => {
    return deg2rad(FOV_Y);
  }

  handleMouseDown = (e: MouseEvent) => {
    // set data
    const loc = this.getNormalizedMouseCoords(e);
    this.mouseLoc = { start: loc, current: loc, previous: loc };
  }

  handleScroll = (e: WheelEvent) => {
    this.offsetRadius += e.deltaY / 50;
    e.preventDefault();
  }

  handleMouseMove = (e: MouseEvent) => {
    if (this.mouseLoc === null) {
      return;
    }

    // set mouse locations
    this.mouseLoc.previous = this.mouseLoc.current
    this.mouseLoc.current = this.getNormalizedMouseCoords(e);

    const a = projectTrackball(this.mouseLoc.start);
    const b = projectTrackball(this.mouseLoc.current);

    vec3.normalize(a, a);
    vec3.normalize(b, b);

    // quaternion rotation between these vectors
    quat.rotationTo(this.currQ, a, b);

  }

  handleMouseUp = (e: MouseEvent) => {
    if (this.mouseLoc === null) {
      return;
    }

    // save the rotational diff between the last pos and the current one
    const a = projectTrackball(this.mouseLoc.previous);
    const b = projectTrackball(this.mouseLoc.current);

    vec3.normalize(a, a);
    vec3.normalize(b, b);

    quat.rotationTo(this.momentumQ, a, b);

    // commit the quaternion change
    quat.mul(this.baseQ, this.currQ, this.baseQ);
    this.currQ = quat.create();

    // mouse up
    this.mouseLoc = null;

    // set momentum level to 1
    this.currentMomentumLevel = 1;
  }

  discardTouchEvent = (e: TouchEvent) => e.preventDefault();

  constructor(ctx: HTMLElement, options: TrackballCameraOptions) {
    if (options.rotation) {
      this.rotationQ = options.rotation;
    } else {
      this.rotationQ = quat.create();
    }

    if(options.dampingFactor !== undefined) {
        this.dampingFactor = options.dampingFactor;
    } else {
        this.dampingFactor = 0.9;
    }


    this.canvas = ctx;

    this.canvas.addEventListener('pointerdown', this.handleMouseDown);
    this.canvas.addEventListener('wheel', this.handleScroll);
    window.addEventListener('pointermove', this.handleMouseMove);
    window.addEventListener('pointerup', this.handleMouseUp);

    // disable touch movements
    this.canvas.addEventListener("touchstart",  this.discardTouchEvent)
    this.canvas.addEventListener("touchmove",   this.discardTouchEvent)
    this.canvas.addEventListener("touchend",    this.discardTouchEvent)
    this.canvas.addEventListener("touchcancel", this.discardTouchEvent)
  }


  cleanup = () => {
    this.canvas.removeEventListener('pointerdown', this.handleMouseDown);
    this.canvas.removeEventListener('wheel', this.handleScroll);
    window.removeEventListener('pointermove', this.handleMouseMove);
    window.removeEventListener('pointerup', this.handleMouseUp);

    // reenable touch movements
    this.canvas.removeEventListener("touchstart",  this.discardTouchEvent)
    this.canvas.removeEventListener("touchmove",   this.discardTouchEvent)
    this.canvas.removeEventListener("touchend",    this.discardTouchEvent)
    this.canvas.removeEventListener("touchcancel", this.discardTouchEvent)
  }


  update = () => {
    if (this.mouseLoc === null) {
      const combinedQ = quat.slerp(quat.create(), this.rotationQ, this.momentumQ, this.currentMomentumLevel);
      this.currentMomentumLevel *= this.dampingFactor;
      quat.mul(this.baseQ, combinedQ , this.baseQ);
    }  
  }

  viewMatrix = () => {
    const tmp = quat.mul(quat.create(), this.currQ, this.baseQ);

    const view = mat4.fromRotationTranslation(mat4.create(), tmp, [0, 0, this.offsetRadius]);
   
    return view;
  }

  viewProjMatrix = (width: number, height: number) => {
    const projMatrix = mat4.perspective(
      mat4.create(),
      this.fov(),
      width / height,
      1,
      1000
    );

    const viewMatrix = this.viewMatrix();
    return mat4.mul(mat4.create(), projMatrix, viewMatrix);
  }
}


