'use client';

import classNames from 'classnames';
import { Menu } from 'lucide-react';
import { toggleSidebar } from './Sidebar';

interface Props {
  className?: string;
  children?: React.ReactNode;
}

export const TopBar: React.FC<Props> = ({ children, className }) => {
  return (
    <div
      className={classNames(
        'absolute top-0 left-0 w-full h-12 bg-gray-900 shadow-sm z-10 flex items-center px-2',
        className,
      )}
    >
      <button
        className="md:hidden p-1 mr-1 text-gray-300 hover:text-white"
        onClick={toggleSidebar}
      >
        <Menu className="w-5 h-5" />
      </button>
      {children ? children : null}
    </div>
  );
};

export const MainContent: React.FC<Props> = ({ children, className }) => {
  return (
    <div className={classNames('pt-14 px-4 absolute top-0 left-0 w-full h-full overflow-auto', className)}>
      {children ? children : null}
    </div>
  );
};
