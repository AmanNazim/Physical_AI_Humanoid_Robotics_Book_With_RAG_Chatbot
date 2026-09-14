import type { ReactNode } from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import ModuleJourney from "@site/src/components/ModuleJourney";
import StudyCompanion from "@site/src/components/StudyCompanion";
import Heading from "@theme/Heading";

import styles from "./index.module.css";

/**
 * Hero — the front door of the book.
 * Blueprint grid + a single oscilloscope trace (the Signal Path, the
 * site's signature): a robot's nervous system carrying a pulse.
 */
function HomepageHero() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <svg
        className={styles.signalTrace}
        viewBox="0 0 1440 220"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <path
          className={styles.tracePath}
          d="M0 70 H430 l14 -34 22 68 14 -34 H960 l14 -34 22 68 14 -34 H1440"
        />
      </svg>
      <div className="container">
        <p className={styles.eyebrow}>For curious beginners</p>
        <Heading as="h1" className={styles.heroTitle}>
          Robots that learn by doing.
          <br />
          You will, too.
        </Heading>
        <p className={styles.heroSubtitle}>
          A four-module path from your first ROS 2 node to robots that see,
          listen, and act — no robotics degree required.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/preface/"
          >
            Start with the preface
          </Link>
          <Link
            className={clsx("button button--lg", styles.secondaryButton)}
            to="#ask-the-book"
          >
            Meet your study companion
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome | ${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics with RAG Chatbot - An interactive educational resource with AI-powered learning assistance"
    >
      <HomepageHero />
      <main>
        <ModuleJourney />
        <StudyCompanion />
      </main>
    </Layout>
  );
}
