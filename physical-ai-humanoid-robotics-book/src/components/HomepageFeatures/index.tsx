import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Physical AI Foundations',
    description: (
      <>
        Learn the fundamentals of Physical AI - where artificial intelligence meets the real world through embodied experience.
        Understand how robots learn through interaction with their environment.
      </>
    ),
  },
  {
    title: 'Humanoid Robotics',
    description: (
      <>
        Explore the cutting-edge field of humanoid robotics, including bipedal locomotion, manipulation, and human-robot interaction.
        Master the challenges of creating robots that operate in human environments.
      </>
    ),
  },
  {
    title: 'Complete Learning Path',
    description: (
      <>
        From ROS 2 fundamentals to Vision-Language-Action systems, follow a structured curriculum that builds expertise
        in all aspects of physical AI and humanoid robotics development.
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

export default function HomepageFeatures(): ReactNode {
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
