document.addEventListener("DOMContentLoaded", function () {
    console.log("✅ projects.js loaded");

    const projectsData = [
        { title: "Predicting Customer Churn", description: "Machine learning for telecom retention strategies.", link: "projects.html?id=project1" },
        { title: "Data Scraper", description: "A web scraper for extracting data from e-commerce websites.", link: "projects.html?id=project2" },
        { title: "AI Chatbot", description: "Conversational AI chatbot for automating customer service.", link: "projects.html?id=project3" },
        { title: "Stock Market Predictor", description: "AI-based predictive analytics for stock trends.", link: "projects.html?id=project4" }
    ];

    const projectsContainer = document.getElementById("projects-container");
    const loadMoreBtn = document.getElementById("load-more-btn");

    let visibleProjects = 2;

    function displayProjects() {
        projectsContainer.innerHTML = "";
        projectsData.slice(0, visibleProjects).forEach((project) => {
            const projectCard = document.createElement("div");
            projectCard.className = "bg-white shadow-lg rounded-lg p-6 transition-transform transform hover:scale-105";
            projectCard.innerHTML = `
                <h3 class="text-2xl font-semibold text-gray-800 mb-2">${project.title}</h3>
                <p class="text-gray-600 mb-4">${project.description}</p>
                <a href="${project.link}"
                   class="inline-block bg-blue-500 text-white font-medium px-5 py-2 rounded-lg hover:bg-blue-600 transition-all">
                   View Project
                </a>
            `;
            projectsContainer.appendChild(projectCard);
        });

        if (visibleProjects >= projectsData.length) {
            loadMoreBtn.style.display = "none";
        }
    }

    loadMoreBtn.addEventListener("click", () => {
        visibleProjects += 2;
        displayProjects();
    });

    displayProjects();
});
