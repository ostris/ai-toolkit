import { Disclosure, DisclosureButton, DisclosurePanel } from '@headlessui/react';
import { FaChevronDown } from 'react-icons/fa';
import classNames from 'classnames';

interface CardProps {
  title?: string;
  children?: React.ReactNode;
  collapsible?: boolean;
  defaultOpen?: boolean;
}

const Card: React.FC<CardProps> = ({ title, children, collapsible, defaultOpen }) => {
  if (collapsible) {
    return (
      <Disclosure as="section" className="space-y-2 px-4 pb-2 pt-2 bg-gray-900 rounded-lg" defaultOpen={defaultOpen}>
        {({ open }) => (
          <>
            <DisclosureButton className="w-full text-left flex items-center justify-between">
              <div className="flex-1">
                {title && (
                  <h2 className={classNames('text-lg mb-2 font-semibold uppercase text-gray-500', { 'mb-0': !open })}>
                    {title}
                  </h2>
                )}
              </div>
              <FaChevronDown className={`ml-2 inline-block transition-transform ${open ? 'rotate-180' : ''}`} />
            </DisclosureButton>
            <DisclosurePanel>{children ?? null}</DisclosurePanel>
            {open && <div className="pt-2"></div>}
          </>
        )}
      </Disclosure>
    );
  }
  return (
    <section className="space-y-2 px-4 pb-4 pt-2 bg-gray-900 rounded-lg">
      {title && <h2 className="text-lg mb-2 font-semibold uppercase text-gray-500">{title}</h2>}
      {children ?? null}
    </section>
  );
};

export default Card;
