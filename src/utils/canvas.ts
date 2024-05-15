import { vec2 } from "gl-matrix";

export class CanvasMouseTracker {
  private canvas: HTMLCanvasElement;

  // mouse status
  public mousePos: { current: vec2, previous: vec2 } | null = null;

  mouseDownListeners: Array<(p: vec2) => void> = [];
  mouseDragListeners: Array<(p: vec2) => void> = [];
  mouseMoveListeners: Array<(p: vec2) => void> = [];
  mouseUpListeners: Array<(p: vec2) => void> = [];
  mouseClickListeners: Array<(p: vec2) => void> = [];
  keyDownListeners: Array<(e: string) => void> = [];

  constructor(ctx: HTMLCanvasElement) {
    this.canvas = ctx;

    // add canvas handler
    this.canvas.addEventListener('pointerdown', this.handleMouseDown);
    this.canvas.addEventListener('pointermove', this.handleMouseDrag);
    window.addEventListener('pointerup', this.handleMouseUp);
    this.canvas.addEventListener('dblclick', this.handleMouseClick);
    this.canvas.addEventListener('keydown', this.handleKeyDown);
    // disable touch movements
    this.canvas.addEventListener("touchstart", this.discardTouchEvent)
    this.canvas.addEventListener("touchmove", this.discardTouchEvent)
    this.canvas.addEventListener("touchend", this.discardTouchEvent)
    this.canvas.addEventListener("touchcancel", this.discardTouchEvent)
  }

  addMouseDownListener = (f: (p: vec2) => void) => { this.mouseDownListeners.push(f) };
  removeMouseDownListener = (f: (p: vec2) => void) => { this.mouseDownListeners = this.mouseDownListeners.filter(x => x !== f) };
  addMouseDragListener = (f: (p: vec2) => void) => { this.mouseDragListeners.push(f) };
  removeMouseDragListener = (f: (p: vec2) => void) => { this.mouseDragListeners = this.mouseDragListeners.filter(x => x !== f) };
  addMouseMoveListener = (f: (p: vec2) => void) => { this.mouseMoveListeners.push(f) };
  removeMouseMoveListener = (f: (p: vec2) => void) => { this.mouseMoveListeners = this.mouseMoveListeners.filter(x => x !== f) };
  addMouseUpListener = (f: (p: vec2) => void) => { this.mouseUpListeners.push(f) };
  removeMouseUpListener = (f: (p: vec2) => void) => { this.mouseUpListeners = this.mouseUpListeners.filter(x => x !== f) };
  addMouseClickListener = (f: (p: vec2) => void) => { this.mouseClickListeners.push(f) };
  removeMouseClickListener = (f: (p: vec2) => void) => { this.mouseClickListeners = this.mouseClickListeners.filter(x => x !== f) };
  addKeyDownListener = (f: (s: string) => void) => { this.keyDownListeners.push(f) };
  removeKeyDownListener = (f: (s: string) => void) => { this.keyDownListeners = this.keyDownListeners.filter(x => x !== f) };

  private getMousePos(canvas: HTMLCanvasElement, evt: MouseEvent): vec2 {
    const rect = canvas.getBoundingClientRect(); // abs. size of element
    const scaleX = canvas.width / rect.width;    // relationship bitmap vs. element for X
    const scaleY = canvas.height / rect.height;  // relationship bitmap vs. element for Y

    return [
      (evt.clientX - rect.left) * scaleX,   // scale mouse coordinates after they have
      (evt.clientY - rect.top) * scaleY     // been adjusted to be relative to element
    ]
  }

  handleMouseDown = (e: MouseEvent) => {
    const v = this.getMousePos(this.canvas, e);
    this.mousePos = {
      current: v,
      previous: v
    };
    for (const f of this.mouseDownListeners) {
      f(v);
    }
  }
  handleMouseUp = (e: MouseEvent) => {
    this.mousePos = null;
    const v = this.getMousePos(this.canvas, e);
    for (const f of this.mouseUpListeners) {
      f(v);
    }
  }

  handleMouseDrag = (e: MouseEvent) => {
    const v = this.getMousePos(this.canvas, e);
    for (const f of this.mouseMoveListeners) {
      f(v);
    }
    if (!this.mousePos) {
      return;
    }
    this.mousePos = {
      current: v,
      previous: this.mousePos.current
    };
    for (const f of this.mouseDragListeners) {
      f(v);
    }
  }

  handleMouseClick = (e: MouseEvent) => {
    const v = this.getMousePos(this.canvas, e);
    for (const f of this.mouseClickListeners) {
      f(v);
    }
  }

  handleKeyDown = (e: KeyboardEvent) => {
    for (const f of this.keyDownListeners) {
      f(e.key);
    }
  }

  discardTouchEvent = (e: TouchEvent) => e.preventDefault();

  cleanup = () => {
    // remove listeners on canvas
    this.canvas.removeEventListener('pointerdown', this.handleMouseDown);
    this.canvas.removeEventListener('pointermove', this.handleMouseDrag);
    window.removeEventListener('pointerup', this.handleMouseUp);
    this.canvas.removeEventListener('dblclick', this.handleMouseClick);
    this.canvas.removeEventListener('keydown', this.handleKeyDown);
    // reenable touch movements
    this.canvas.removeEventListener("touchstart", this.discardTouchEvent)
    this.canvas.removeEventListener("touchmove", this.discardTouchEvent)
    this.canvas.removeEventListener("touchend", this.discardTouchEvent)
    this.canvas.removeEventListener("touchcancel", this.discardTouchEvent)
  }
}
