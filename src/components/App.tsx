import { useEffect } from 'react';
import { DEPTHS } from '../types';
import { useView } from '../state/store';
import { Canvas } from './Canvas';
import { ExplorePanel } from './ExplorePanel';
import { Header } from './Header';
import { Inspector } from './Inspector';
import { Notices } from './Notices';
import { ScenarioBar } from './ScenarioBar';

/** Keys 1 / 2 / 3 set the depth, unless the user is typing in a field. */
function useDepthShortcuts() {
  const setDepth = useView((s) => s.setDepth);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target;
      if (t instanceof HTMLElement && (t.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName))) return;
      const d = DEPTHS.find((x) => String(x) === e.key);
      if (d) setDepth(d);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setDepth]);
}

export function App() {
  useDepthShortcuts();
  return (
    <div className="flex min-h-dvh flex-col lg:h-dvh">
      <Header />
      <Notices />
      <main className="grid flex-1 grid-cols-1 lg:min-h-0 lg:grid-cols-[15rem_minmax(0,1fr)_22rem]">
        <ExplorePanel />
        <Canvas />
        <Inspector />
      </main>
      <ScenarioBar />
    </div>
  );
}
