import { DepthControl } from './DepthControl';
import { LensTabs } from './LensTabs';
import { ModelSelector } from './ModelSelector';

export function Header() {
  return (
    <header className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-line bg-panel px-4 py-2">
      <h1 className="font-mono text-sm font-semibold tracking-[0.3em] text-accent">AI-FLOW</h1>
      <ModelSelector />
      {/* Below lg the tabs take their own full-width row and scroll sideways instead of wrapping. */}
      <nav aria-label="Lenses" className="order-last w-full min-w-0 lg:order-none lg:w-auto lg:flex-1">
        <LensTabs />
      </nav>
      <DepthControl />
    </header>
  );
}
