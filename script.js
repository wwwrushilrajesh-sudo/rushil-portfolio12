const menuToggle = document.querySelector('.menu-toggle');
const nav = document.querySelector('.nav-links');

menuToggle?.addEventListener('click', () => {
  const isOpen = nav.classList.toggle('open');
  menuToggle.setAttribute('aria-expanded', String(isOpen));
  menuToggle.textContent = isOpen ? '✕' : '☰';
});

document.querySelectorAll('.nav-links a').forEach(link => {
  link.addEventListener('click', () => {
    nav.classList.remove('open');
    menuToggle?.setAttribute('aria-expanded', 'false');
    if (menuToggle) menuToggle.textContent = '☰';
  });
});

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.08 });

document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

document.getElementById('year').textContent = new Date().getFullYear();


const projectTabs = document.querySelectorAll('[data-project-tab]');
const projectPanels = document.querySelectorAll('.project-tab-panel');

function activateProjectTab(id, updateHash = false) {
  projectTabs.forEach(tab => {
    const active = tab.dataset.projectTab === id;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  projectPanels.forEach(panel => {
    const active = panel.id === 'project-panel-' + id;
    panel.classList.toggle('active', active);
    panel.hidden = !active;
  });
  if (updateHash) history.replaceState(null, '', '#project-' + id);
}

projectTabs.forEach(tab => {
  tab.addEventListener('click', () => activateProjectTab(tab.dataset.projectTab, true));
  tab.addEventListener('keydown', e => {
    if (!['ArrowDown','ArrowUp','ArrowRight','ArrowLeft'].includes(e.key)) return;
    e.preventDefault();
    const list = [...projectTabs];
    const current = list.indexOf(tab);
    const delta = (e.key === 'ArrowDown' || e.key === 'ArrowRight') ? 1 : -1;
    const next = list[(current + delta + list.length) % list.length];
    next.focus();
    activateProjectTab(next.dataset.projectTab, true);
  });
});

const projectHash = location.hash.match(/^#project-(.+)$/);
activateProjectTab(projectHash ? projectHash[1] : 'aerospike');
