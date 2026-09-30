import { createContext, useContext, type ReactNode } from 'react';
import { createStore, useStore } from 'zustand';
import type { Depth, LensId, ModelSpec, Scenario } from '../types';
import { getModel } from '../data/registry';
import type { ViewState } from './url';

export interface ViewActions {
  /** Keeps the lens, depth and selection. */
  setModel(modelId: string): void;
  /** Keeps the model, depth and selection. */
  setLens(lens: LensId): void;
  /** Expands in place: keeps the selection. */
  setDepth(depth: Depth): void;
  selectNode(nodeId: string | null): void;
  patchScenario(patch: Partial<Scenario>): void;
  /** Replace the whole view (browser back / forward). */
  load(view: ViewState): void;
  dismissNotices(): void;
}

export type ViewStoreState = ViewState & ViewActions & { notices: readonly string[] };

export function createViewStore(initial: ViewState, notices: readonly string[] = []) {
  return createStore<ViewStoreState>()((set) => ({
    ...initial,
    notices,
    setModel: (modelId) => set({ modelId }),
    setLens: (lens) => set({ lens }),
    setDepth: (depth) => set({ depth }),
    selectNode: (nodeId) => set({ nodeId }),
    patchScenario: (patch) => set((s) => ({ scenario: { ...s.scenario, ...patch } })),
    load: (view) => set(view),
    dismissNotices: () => set({ notices: [] }),
  }));
}

export type ViewStore = ReturnType<typeof createViewStore>;

export function pickView(s: ViewState): ViewState {
  return { modelId: s.modelId, lens: s.lens, depth: s.depth, nodeId: s.nodeId, scenario: s.scenario };
}

const StoreContext = createContext<ViewStore | null>(null);

export function ViewProvider({ store, children }: { store: ViewStore; children: ReactNode }) {
  return <StoreContext.Provider value={store}>{children}</StoreContext.Provider>;
}

export function useView<T>(selector: (s: ViewStoreState) => T): T {
  const store = useContext(StoreContext);
  if (!store) throw new Error('useView must be used inside <ViewProvider>');
  return useStore(store, selector);
}

export function useModel(): ModelSpec {
  return getModel(useView((s) => s.modelId));
}
