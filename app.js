const copyButtons = document.querySelectorAll("[data-copy-target]");

for (const button of copyButtons) {
  button.addEventListener("click", async () => {
    const target = document.querySelector(button.dataset.copyTarget);
    const text = target ? target.textContent.trim() : "";
    try {
      await navigator.clipboard.writeText(text);
      const defaultText = button.dataset.defaultText || button.textContent;
      button.dataset.defaultText = defaultText;
      button.textContent = "Copied";
      setTimeout(() => {
        button.textContent = defaultText;
      }, 1400);
    } catch {
      button.textContent = "Copy failed";
    }
  });
}

document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", (event) => {
    const target = document.querySelector(anchor.getAttribute("href"));
    if (!target) return;
    event.preventDefault();
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
});
