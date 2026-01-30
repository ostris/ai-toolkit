import classNames from 'classnames';
import { FaBars } from "react-icons/fa";

interface Props {
  className?: string;
  children?: React.ReactNode;
}

export const TopBar: React.FC<Props> = ({ children, className }) => {
  return (
    <div
      className={classNames(
        'absolute top-0 left-0 w-full h-12  dark:bg-gray-900 shadow-sm z-10 flex items-center px-2',
        className,
      )}
    >
      <label className="mr-2 cursor-pointer sm:hidden">
        <input id="sidebar-toggle" type="checkbox" hidden />
        <FaBars />
      </label>

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
