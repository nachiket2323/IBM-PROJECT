/* ============================================
   Book Recommendation System — Client JS
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initGenreFilter();
});

/* ---------- Tab Switching ---------- */
function initTabs() {
  const buttons = document.querySelectorAll('.tab-btn');
  const panels  = document.querySelectorAll('.tab-panel');

  if (!buttons.length) return;

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;

      buttons.forEach(b => b.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const panel = document.getElementById('tab-' + target);
      if (panel) {
        panel.classList.add('active');

        // Lazy-load content if panel is empty and has a data-url
        if (panel.dataset.url && !panel.dataset.loaded) {
          loadTabContent(panel);
        }
      }
    });
  });

  // Activate first tab
  if (buttons[0]) buttons[0].click();
}

function loadTabContent(panel) {
  panel.innerHTML = '<div class="spinner">Loading recommendations…</div>';

  fetch(panel.dataset.url)
    .then(res => res.text())
    .then(html => {
      panel.innerHTML = html;
      panel.dataset.loaded = '1';
    })
    .catch(() => {
      panel.innerHTML = '<div class="alert alert--error">Failed to load. Please refresh.</div>';
    });
}

/* ---------- Genre Filter ---------- */
function initGenreFilter() {
  document.querySelectorAll('.genre-tag').forEach(tag => {
    tag.addEventListener('click', () => {
      tag.classList.toggle('active');
    });
  });
}

/* ---------- Add Friend ---------- */
function addFriend(event) {
  event.preventDefault();
  const form = event.target;
  const input = form.querySelector('input[name="friend_id"]');
  const friendId = input.value.trim();
  if (!friendId) return;

  fetch('/add_friend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: 'friend_id=' + encodeURIComponent(friendId)
  })
  .then(res => res.json())
  .then(data => {
    if (data.ok) {
      // Reload to reflect changes
      location.reload();
    } else {
      alert(data.message || 'Could not add friend');
    }
  })
  .catch(() => alert('Network error'));
}
