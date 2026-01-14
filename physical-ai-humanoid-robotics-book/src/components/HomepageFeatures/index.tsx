import React from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Educational Resource',
    description: (
      <>
        Comprehensive guide to Physical AI & Humanoid Robotics concepts,
        from fundamentals to advanced implementations.
      </>
    ),
  },
  {
    title: 'Practical Implementation',
    description: (
      <>
        Real-world examples and hands-on projects using ROS 2, AI frameworks,
        and robotics platforms.
      </>
    ),
  },
  {
    title: 'Cutting-Edge Technology',
    description: (
      <>
        Coverage of latest developments in humanoid robotics, embodied AI,
        and vision-language-action systems.
      </>
    ),
  },
];

function Feature({title, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className={clsx('text--center padding-horiz--md', styles.featureCard)}>
        <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
        <p className={styles.featureDescription}>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}