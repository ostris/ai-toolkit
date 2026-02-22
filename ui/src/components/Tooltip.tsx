import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';

interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

export const Tooltip: React.FC<TooltipProps> = ({ content, children, className = '' }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const tooltipId = useRef(`tooltip-${Math.random().toString(36).substr(2, 9)}`);

  useEffect(() => {
    if (isVisible && triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      const tooltipWidth = tooltipRef.current?.offsetWidth || 200;
      
      // Position tooltip above the trigger element, centered
      const left = rect.left + rect.width / 2 - tooltipWidth / 2;
      const top = rect.top - 10; // 10px above the trigger

      setPosition({
        top: Math.max(10, top), // Ensure it doesn't go off-screen at top
        left: Math.max(10, Math.min(left, window.innerWidth - tooltipWidth - 10)), // Keep within viewport
      });
    }
    
    return () => {
      // Cleanup function for safe unmounting
    };
  }, [isVisible]);

  const tooltipContent = isVisible && content ? (
    <div
      ref={tooltipRef}
      id={tooltipId.current}
      role="tooltip"
      className="fixed z-50 px-3 py-2 text-sm text-white bg-gray-900 rounded-lg shadow-lg border border-gray-700 max-w-xs"
      style={{
        top: `${position.top}px`,
        left: `${position.left}px`,
        transform: 'translateY(-100%)',
        pointerEvents: 'none',
      }}
    >
      {content}
      {/* Arrow pointing down */}
      <div
        className="absolute w-2 h-2 bg-gray-900 border-r border-b border-gray-700"
        style={{
          bottom: '-5px',
          left: '50%',
          transform: 'translateX(-50%) rotate(45deg)',
        }}
      />
    </div>
  ) : null;

  return (
    <>
      <div
        ref={triggerRef}
        className={`inline-block ${className}`}
        tabIndex={0}
        aria-describedby={isVisible ? tooltipId.current : undefined}
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
      >
        {children}
      </div>
      {typeof window !== 'undefined' && tooltipContent && createPortal(tooltipContent, document.body)}
    </>
  );
};
