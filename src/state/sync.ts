import { pickView, type ViewStore } from './store';
import { parseView, serializeView } from './url';

/**
 * Keeps the store and the address bar in lockstep:
 * - on start, rewrites the URL to its canonical form (fallbacks applied);
 * - model / lens / node / depth changes push a history entry, so Back works;
 * - scenario changes replace the current entry, so dragging a slider does not flood history;
 * - Back / Forward reload the view from the URL.
 *
 * Returns an unsubscribe function.
 */
export function syncWithUrl(store: ViewStore, win: Window): () => void {
  const write = (push: boolean) => {
    const search = serializeView(pickView(store.getState()));
    if (search === win.location.search) return;
    const url = `${win.location.pathname}${search}${win.location.hash}`;
    try {
      if (push) win.history.pushState(null, '', url);
      else win.history.replaceState(null, '', url);
    } catch {
      // Some browsers refuse history changes on file:// pages. The view still works; only the link does not update.
    }
  };

  write(false);
  const unsubscribe = store.subscribe((s, prev) => {
    const navigated = s.modelId !== prev.modelId || s.lens !== prev.lens || s.depth !== prev.depth || s.nodeId !== prev.nodeId;
    if (navigated || s.scenario !== prev.scenario) write(navigated);
  });
  const onPop = () => store.getState().load(parseView(win.location.search).view);
  win.addEventListener('popstate', onPop);

  return () => {
    unsubscribe();
    win.removeEventListener('popstate', onPop);
  };
}
