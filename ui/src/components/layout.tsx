import classNames from 'classnames';

interface Props {
  className?: string;
  children?: React.ReactNode;
}

export const TopBar: React.FC<Props> = ({ children, className }) => {
  return (
    <div
      className={classNames(
        'absolute top-0 left-0 w-full h-12 dark:bg-gray-900 shadow-sm z-10 flex items-center px-2 pl-12 md:pl-2',
        className,
      )}
    >
      {children ? children : null}
    </div>
  );
};

export const MainContent: React.FC<Props> = ({ children, className }) => {
  return (
    <div className={classNames('pt-14 pb-8 px-1 md:px-4 absolute top-0 left-0 right-0 h-full overflow-y-auto overflow-x-hidden', className)}>
      {children ? children : null}
    </div>
  );
};
