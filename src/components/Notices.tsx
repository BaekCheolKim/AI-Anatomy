import { useView } from '../state/store';

/** Explains anything in a pasted link that could not be honoured. */
export function Notices() {
  const notices = useView((s) => s.notices);
  const dismiss = useView((s) => s.dismissNotices);
  if (notices.length === 0) return null;
  return (
    <div role="status" className="flex items-start gap-4 border-b border-line bg-raised px-4 py-2 text-sm">
      <ul className="flex-1 text-warn">
        {notices.map((n) => (
          <li key={n}>{n}</li>
        ))}
      </ul>
      <button type="button" onClick={dismiss} className="rounded px-2 text-muted hover:text-fg">
        Dismiss
      </button>
    </div>
  );
}
