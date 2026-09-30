// @vitest-environment jsdom
import { act, cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it } from 'vitest';
import { createViewStore, ViewProvider } from '../state/store';
import { syncWithUrl } from '../state/sync';
import { parseView } from '../state/url';
import { App } from './App';

let stopSync = () => {};

function mount(search: string) {
  window.history.replaceState(null, '', `/${search}`);
  const { view, notices } = parseView(window.location.search);
  const store = createViewStore(view, notices);
  stopSync = syncWithUrl(store, window);
  render(
    <ViewProvider store={store}>
      <App />
    </ViewProvider>,
  );
  return store;
}

afterEach(() => {
  stopSync();
  cleanup();
});

const tab = (name: string) => screen.getByRole('tab', { name });
const depthRadio = (d: number) => screen.getByRole('radio', { name: `Depth ${d}` });
const isOn = (el: HTMLElement, attr: 'aria-selected' | 'aria-checked') => el.getAttribute(attr) === 'true';

describe('URL is the source of truth', () => {
  it('restores lens, depth and scenario from a pasted link, then writes the canonical form', () => {
    mount('?lens=memory&depth=2&b=16');
    expect(isOn(tab('Memory'), 'aria-selected')).toBe(true);
    expect(isOn(depthRadio(2), 'aria-checked')).toBe(true);
    expect(screen.getByText('16')).toBeTruthy();
    expect(window.location.search).toMatch(/^\?model=[^&]+&lens=memory&depth=2&b=16$/);
  });

  it('switching lens keeps depth and pushes one history entry', async () => {
    mount('?depth=3');
    const before = window.history.length;
    await userEvent.setup().click(tab('Inference'));
    expect(window.location.search).toContain('lens=inference');
    expect(window.location.search).toContain('depth=3');
    expect(window.history.length).toBe(before + 1);
  });

  it('scenario changes replace the history entry instead of pushing', () => {
    const store = mount('');
    const before = window.history.length;
    act(() => store.getState().patchScenario({ batch: 32 }));
    expect(window.location.search).toContain('b=32');
    expect(window.history.length).toBe(before);
  });

  it('Back restores the previous view', async () => {
    mount('?lens=training');
    await userEvent.setup().click(tab('HPC'));
    expect(isOn(tab('HPC'), 'aria-selected')).toBe(true);
    await act(async () => {
      const popped = new Promise((r) => window.addEventListener('popstate', r, { once: true }));
      window.history.back();
      await popped;
    });
    expect(isOn(tab('Training'), 'aria-selected')).toBe(true);
    expect(window.location.search).toContain('lens=training');
  });

  it('explains values it could not use, and can be dismissed', async () => {
    mount('?model=no-such-model&depth=9');
    const status = screen.getByRole('status');
    expect(within(status).getByText(/Unknown model "no-such-model"/)).toBeTruthy();
    expect(within(status).getByText(/Ignored depth=9/)).toBeTruthy();
    await userEvent.setup().click(within(status).getByRole('button', { name: 'Dismiss' }));
    expect(screen.queryByRole('status')).toBeNull();
  });
});

describe('keyboard', () => {
  it('arrow, Home and End keys move between lens tabs', async () => {
    mount('');
    const user = userEvent.setup();
    tab('Architecture').focus();
    await user.keyboard('{ArrowRight}');
    expect(isOn(tab('Inference'), 'aria-selected')).toBe(true);
    expect(document.activeElement).toBe(tab('Inference'));
    await user.keyboard('{End}');
    expect(isOn(tab('HPC'), 'aria-selected')).toBe(true);
    await user.keyboard('{ArrowRight}');
    expect(isOn(tab('Architecture'), 'aria-selected')).toBe(true);
  });

  it('1 / 2 / 3 set the depth, but not while a form control has focus', async () => {
    mount('');
    const user = userEvent.setup();
    await user.keyboard('3');
    expect(isOn(depthRadio(3), 'aria-checked')).toBe(true);
    screen.getByRole('combobox').focus();
    await user.keyboard('2');
    expect(isOn(depthRadio(3), 'aria-checked')).toBe(true);
  });

  it('arrow keys move within the depth control', async () => {
    mount('');
    const user = userEvent.setup();
    depthRadio(1).focus();
    await user.keyboard('{ArrowRight}{ArrowRight}');
    expect(isOn(depthRadio(3), 'aria-checked')).toBe(true);
    expect(document.activeElement).toBe(depthRadio(3));
  });
});

describe('computed values', () => {
  it('shows the model card with computed parameter counts', () => {
    mount('');
    expect(screen.getAllByText('8.03 B').length).toBeGreaterThan(0);
    expect(screen.getByText(/32 query \/ 8 KV heads/)).toBeTruthy();
    // training tokens are flagged approximate in the model's sources
    expect(screen.getAllByText('approx').length).toBeGreaterThan(0);
  });

  it('scenario bar updates HBM when the batch changes', () => {
    const store = mount('');
    const meter = screen.getByRole('meter');
    const used = Number(meter.getAttribute('aria-valuenow'));
    act(() => store.getState().patchScenario({ batch: 16 }));
    expect(Number(screen.getByRole('meter').getAttribute('aria-valuenow'))).toBeGreaterThan(used);
  });

  it('flags OOM when the scenario does not fit', () => {
    mount('?gpu=rtx4090');
    expect(screen.getByText('OOM')).toBeTruthy();
  });
});
