import { useId, type ReactNode } from 'react';

export function Panel({ title, className = '', children }: { title: string; className?: string; children: ReactNode }) {
  const id = useId();
  return (
    <section aria-labelledby={id} className={`flex min-h-0 flex-col bg-panel ${className}`}>
      <h2 id={id} className="panel-label border-b border-line px-4 py-2">
        {title}
      </h2>
      <div className="min-h-0 flex-1 overflow-auto p-4">{children}</div>
    </section>
  );
}
