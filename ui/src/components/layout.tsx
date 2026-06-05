'use client';
import React from 'react';
import classNames from 'classnames';
import ThemeLogo from './ThemeLogo';
import { mobileSidebarState } from './Sidebar';

interface Props {
  className?: string;
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

export const MainContent = React.forwardRef<HTMLDivElement, Props>(({ children, className }, ref) => {
  return (
    <div
      ref={ref}
      className={classNames('pt-14 px-2 sm:px-4 absolute top-0 left-0 w-full h-full overflow-auto', className)}
    >
      {children ? children : null}
    </div>
  );
});
MainContent.displayName = 'MainContent';
