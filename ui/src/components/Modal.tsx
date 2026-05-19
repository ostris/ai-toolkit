import React, { Fragment, useEffect } from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  showCloseButton?: boolean;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  closeOnOverlayClick?: boolean;
}

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
  showCloseButton = true,
  size = 'md',
  closeOnOverlayClick = true,
}) => {
  // Close on ESC key press
  useEffect(() => {
    const handleEscKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscKey);
      // Prevent body scrolling when modal is open
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscKey);
      document.body.style.overflow = 'auto';
    };
  }, [isOpen, onClose]);

  // Handle overlay click
  const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget && closeOnOverlayClick) {
      onClose();
    }
  };

  if (!isOpen) return null;

  // Size mapping
  const sizeClasses = {
    sm: 'max-w-md',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
    xl: 'max-w-4xl',
  };

  return (
    <Fragment>
      {/* Modal backdrop */}
      <div
        className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto bg-gray-900 bg-opacity-75 backdrop-blur-sm transition-opacity p-2 sm:p-4"
        onClick={handleOverlayClick}
        aria-modal="true"
        role="dialog"
        aria-labelledby="modal-title"
      >
        {/* Modal panel */}
        <div
          className={`relative mx-auto w-full ${sizeClasses[size]} max-h-full overflow-y-auto rounded-xl bg-gray-900 shadow-xl shadow-black/10 dark:shadow-2xl dark:shadow-black/80 transition-all`}
          onClick={e => e.stopPropagation()}
        >
          {/* Modal header */}
          {(title || showCloseButton) && (
            <div className="flex items-center justify-between bg-gray-800 px-4 py-3 rounded-t-xl">
              {title && (
                <h3 id="modal-title" className="font-semibold text-gray-100 truncate pr-2">
                  {title}
                </h3>
              )}

              {showCloseButton && (
                <button
                  type="button"
                  className="ml-auto inline-flex items-center justify-center rounded-md p-2 text-gray-400 hover:bg-gray-700 hover:text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 flex-shrink-0"
                  onClick={onClose}
                  aria-label="Close modal"
                >
                  <svg
                    className="h-5 w-5"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              )}
            </div>
          )}

          {/* Modal content */}
          <div className="px-4 sm:px-6 py-4">{children}</div>
        </div>
      </div>
    </Fragment>
  );
};
