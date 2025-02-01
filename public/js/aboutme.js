document.addEventListener("DOMContentLoaded", () => {
    const highlightItems = document.querySelectorAll(".highlight-item");
    if (highlightItems.length > 0) {
      const observer = new IntersectionObserver(
        (entries, observer) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              entry.target.classList.add("animate");
              observer.unobserve(entry.target);
            }
          });
        },
        { threshold: 0.1 }
      );
  
      highlightItems.forEach((item, index) => {
        item.style.transitionDelay = `${index * 0.2}s`; // Staggered animation
        observer.observe(item);
      });
    }
  
    // ----- Fade Animation for Other Sections -----
    const fadeElements = document.querySelectorAll(".fade-bottom");
    if (fadeElements.length > 0) {
      const observer2 = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("in-view");
          }
        });
      });
  
      fadeElements.forEach((el) => observer2.observe(el));
    }
  });
  
  
  