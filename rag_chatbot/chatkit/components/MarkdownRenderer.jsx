import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeSanitize from 'rehype-sanitize';
import rehypeHighlight from 'rehype-highlight';
import './MarkdownRenderer.css';

const MarkdownRenderer = ({ content }) => {
  // Sanitize the content to prevent XSS
  const sanitizedContent = content || '';

  return (
    <div className="markdown-renderer">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeSanitize, rehypeHighlight]}
        components={{
          // Custom component for code blocks with syntax highlighting
          code(props) {
            const { children, className, ...rest } = props;
            const match = /language-(\w+)/.exec(className || '');

            return match ? (
              <pre {...rest} className={className}>
                <code className={className}>{children}</code>
              </pre>
            ) : (
              <code {...rest} className={className}>{children}</code>
            );
          },
          // Custom component for links to open in new tab
          a(props) {
            const { href, children, ...rest } = props;
            return (
              <a {...rest} href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            );
          },
          // Custom component for tables
          table(props) {
            return (
              <div className="table-container">
                <table {...props} />
              </div>
            );
          }
        }}
      >
        {sanitizedContent}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;