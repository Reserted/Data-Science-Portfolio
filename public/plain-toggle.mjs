// "plain mode" switch, rendered into the top navbar by myst.yml (parts.navbar_end).
// It adds or removes `plain` on <html>; style.css holds the actual look. MyST renders
// anywidgets in a shadow root, so style.css can't reach this button — its CSS lives here.
const CSS = `
.plain-toggle {
  /* same box and type as the LinkedIn action button (Tailwind px-4 py-2 mx-1 leading-none border rounded) */
  font: inherit;
  font-size: 1rem;
  line-height: 1;
  padding: 0.5rem 1rem;
  margin: 0 0.25rem;
  background: transparent;
  border: 1px solid currentColor;
  border-radius: 0.25rem;
  color: inherit;
  opacity: 1;
  cursor: pointer;
}
.plain-toggle:hover { opacity: 0.8; }
/* Pressed = plain mode on. The label stays "Plain"; aria-pressed and the fill carry the state. */
.plain-toggle[aria-pressed="true"] { background: var(--fg); color: var(--bg); border-color: var(--fg); }
`;
export default {
  render({ el }) {
    const root = document.documentElement;
    const button = document.createElement('button');
    button.className = 'plain-toggle';
    button.title = 'Plain sans-serif styling';
    // A toggle with aria-pressed keeps one label; flipping it to "styled" made the
    // pressed state and the label contradict each other for screen readers.
    button.textContent = 'Plain';

    const paint = () => button.setAttribute('aria-pressed', String(root.classList.contains('plain')));

    button.addEventListener('click', () => {
      const on = !root.classList.contains('plain');
      root.classList.toggle('plain', on);
      try { localStorage.setItem('plain', on ? '1' : '0'); } catch { /* private mode */ }
      paint();
    });

    try { if (localStorage.getItem('plain') === '1') root.classList.add('plain'); } catch { /* private mode */ }
    paint();

    // Client-side navigation and the dark-mode switch both rewrite this class list.
    const observer = new MutationObserver(paint);
    observer.observe(root, { attributes: true, attributeFilter: ['class'] });

    const style = document.createElement('style');
    style.textContent = CSS;
    el.append(style, button);
    return () => observer.disconnect();
  },
};
