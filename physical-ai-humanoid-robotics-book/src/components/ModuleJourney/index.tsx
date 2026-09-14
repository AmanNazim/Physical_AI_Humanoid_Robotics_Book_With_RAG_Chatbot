import React from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

/**
 * The four modules are a true prerequisite sequence — the numbers carry
 * real information, and the connecting line is the Signal Path returning
 * from the hero: one continuous thread from first node to VLA.
 */
const MODULES = [
  {
    number: '01',
    title: 'ROS 2 — the nervous system',
    to: '/docs/module-1/introduction',
    description:
      "Wire up your robot's senses and muscles. Every chapter that follows runs on the communication skills you build here.",
    chapters: 4,
  },
  {
    number: '02',
    title: 'Digital twin — Gazebo & Unity',
    to: '/docs/module-2/introduction',
    description:
      'Rehearse in simulation before touching hardware. Break things for free and learn from every crash.',
    chapters: 4,
  },
  {
    number: '03',
    title: 'The AI brain — NVIDIA Isaac',
    to: '/docs/module-3/introduction',
    description:
      'Give your robot judgment: process what it senses and decide what to do next.',
    chapters: 4,
  },
  {
    number: '04',
    title: 'Vision-Language-Action',
    to: '/docs/module-4/introduction',
    description:
      'Put it all together — robots that understand your words, see their surroundings, and act on both.',
    chapters: 4,
  },
];

export default function ModuleJourney(): React.JSX.Element {
  return (
    <section className={styles.journey}>
      <div className="container">
        <p className={styles.eyebrow}>The journey</p>
        <div className={styles.journeyRow}>
          {MODULES.map((module) => (
            <Link key={module.number} to={module.to} className={styles.moduleCard}>
              <span className={styles.moduleNumber}>{module.number}</span>
              <h3 className={styles.moduleTitle}>{module.title}</h3>
              <p className={styles.moduleDescription}>{module.description}</p>
              <span className={styles.moduleMeta}>
                {module.chapters} chapters
              </span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
