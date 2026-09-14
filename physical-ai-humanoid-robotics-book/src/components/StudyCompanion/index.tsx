import React from 'react';
import styles from './styles.module.css';

/**
 * "Ask the book" — the RAG chatbot callout. The one place the human
 * amber accent is allowed to speak.
 */
export default function StudyCompanion(): React.JSX.Element {
  return (
    <section className={styles.companion}>
      <div className="container">
        <div className={styles.card}>
          <svg
            className={styles.icon}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
          </svg>
          <div>
            <p className={styles.eyebrow}>Ask the book</p>
            <h2 id="ask-the-book" className={styles.title}>
              Stuck at 2 a.m.? Ask the book.
            </h2>
            <p className={styles.body}>
              A chatbot that answers from these pages, not the whole
              internet — and shows you where it read the answer.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
