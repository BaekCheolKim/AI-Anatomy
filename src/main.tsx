import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './components/App';
import { createViewStore, ViewProvider } from './state/store';
import { syncWithUrl } from './state/sync';
import { parseView } from './state/url';
import './index.css';

const { view, notices } = parseView(window.location.search);
const store = createViewStore(view, notices);
syncWithUrl(store, window);

const root = document.getElementById('root');
if (!root) throw new Error('index.html is missing #root');

createRoot(root).render(
  <StrictMode>
    <ViewProvider store={store}>
      <App />
    </ViewProvider>
  </StrictMode>,
);
