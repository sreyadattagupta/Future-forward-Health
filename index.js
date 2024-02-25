document.addEventListener("DOMContentLoaded", function () {
  const navLinks = document.querySelectorAll(".removeUnderLine");

  navLinks.forEach(function (link) {
    link.addEventListener("click", function (event) {
      event.preventDefault();
      const targetSectionId = link.getAttribute("href").substring(1);
      const targetSection = document.getElementById(targetSectionId);

      if (targetSection) {
        window.scrollTo({
          top: targetSection.offsetTop - document.querySelector("nav").offsetHeight,
          behavior: "smooth",
        });
      }
    });
  });
});