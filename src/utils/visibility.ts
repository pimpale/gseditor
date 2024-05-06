export class VisibilityChecker {

  private observer: IntersectionObserver;
  private visiblilityCache: boolean;

  constructor(elm: HTMLElement) {
    this.visiblilityCache = false;
    this.observer = new IntersectionObserver((obs, _) => this.callback(obs[0].isIntersecting), {});
    this.observer.observe(elm);
  }

  private callback(visibilityUpdate: boolean) {
    this.visiblilityCache = visibilityUpdate;
  }

  public isVisible(): boolean {
    return this.visiblilityCache;
  }

  public cleanup() {
    this.observer.disconnect();
  }

}
