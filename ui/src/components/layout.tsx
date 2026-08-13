'use client';
import React from 'react';
import classNames from 'classnames';
import ThemeLogo from './ThemeLogo';
import { mobileSidebarState } from './Sidebar';

interface Props {
  className?: string;
  style?: React.CSSProperties;
  children?: React.ReactNode;
}

const MobileMenuButton: React.FC = () => {
  const [, setIsMobileOpen] = mobileSidebarState.use();
  return (
    <button
      onClick={() => setIsMobileOpen(true)}
      className="md:hidden flex items-center ml-2 mr-1 px-1 py-1 rounded-md hover:bg-gray-800"
      aria-label="Open menu"
    >
      <ThemeLogo />
    </button>
  );
};

export const TopBar: React.FC<Props> = ({ children, className }) => {
  return (
    <div
      className={classNames(
        'absolute top-0 left-0 w-full h-12 bg-gray-900 shadow-sm z-10 flex items-center px-2 overflow-x-auto whitespace-nowrap',
        className,
      )}
    >
      <MobileMenuButton />
      {children ? children : null}
    </div>
  );
};

interface MainContentProps extends Props {
  /** Start the scroll container below the fixed TopBar so the scrollbar doesn't run behind it. */
  belowTopBar?: boolean;
}

export const MainContent = React.forwardRef<HTMLDivElement, MainContentProps>(
  ({ children, className, style, belowTopBar }, ref) => {
    return (
      <div
        ref={ref}
        style={style}
        className={classNames(
          'px-2 sm:px-4 absolute left-0 w-full overflow-auto',
          belowTopBar ? 'pt-2 top-12 bottom-0' : 'pt-14 top-0 h-full',
          className,
        )}
      >
        {children ? children : null}
      </div>
    );
  },
);
MainContent.displayName = 'MainContent';
