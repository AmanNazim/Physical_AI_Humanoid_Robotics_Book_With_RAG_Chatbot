import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: "Physical AI & Humanoid Robotics",
  tagline:
    "A comprehensive educational resource for Physical AI and Humanoid Robotics",
  favicon: "img/physical-ai-logo.png",

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: "https://amannazim.github.io",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/Physical_AI_Humanoid_Robotics_Book_With_RAG_Chatbot/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "AmanNazim", // Usually your GitHub org/user name.
  projectName: "Physical_AI_Humanoid_Robotics_Book_With_RAG_Chatbot", // Usually your repo name.

  onBrokenLinks: "throw",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            "https://github.com/AmanNazim/Physical_AI_Humanoid_Robotics_Book_With_RAG_Chatbot/edit/main/physical-ai-humanoid-robotics-book/",
        },
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: "img/docusaurus-social-card.jpg",
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: "Physical AI & Humanoid Robotics",
      logo: {
        alt: "Physical AI & Humanoid Robotics Book Logo",
        src: "img/physical-ai-logo.png",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "tutorialSidebar",
          position: "left",
          label: "Book",
        },
        {
          href: "https://github.com/AmanNazim/Physical_AI_Humanoid_Robotics_Book_With_RAG_Chatbot",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Book Content",
          items: [
            {
              label: "Preface",
              to: "/docs/preface/",
            },
            {
              label: "Module 1: ROS 2 Nervous System",
              to: "/docs/module-1/introduction",
            },
            {
              label: "Module 2: AI Action System",
              to: "/docs/module-2/introduction",
            },
            {
              label: "Module 3: Humanoid Robot Control",
              to: "/docs/module-3/introduction",
            },
            {
              label: "Module 4: Vision-Language-Action",
              to: "/docs/module-4/introduction",
            },
            {
              label: "Assessments",
              to: "/docs/assessments/",
            },
            {
              label: "Hardware Requirements",
              to: "/docs/Hardware-Requirements/",
            },
          ],
        },
        {
          title: "Resources",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/AmanNazim/Physical_AI_Humanoid_Robotics_Book_With_RAG_Chatbot",
            },
            {
              label: "Contributing",
              href: "https://github.com/AmanNazim/Physical_AI_Humanoid_Robotics_Book_With_RAG_Chatbot/blob/main/CONTRIBUTING.md",
            },
          ],
        },
        {
          title: "More",
          items: [
            {
              label: "Physical AI",
              href: "https://en.wikipedia.org/wiki/Physical_artificial_intelligence",
            },
            {
              label: "ROS 2",
              href: "https://docs.ros.org/en/humble/",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Book. Built by Aman Nazim.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
