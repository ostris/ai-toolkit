import { MoreVertical } from 'lucide-react';
import { Menu, MenuButton, MenuItems } from '@headlessui/react';

interface OverflowMenuProps {
  children: React.ReactNode;
  onClick?: (e: React.MouseEvent) => void;
}

export default function OverflowMenu({ children, onClick }: OverflowMenuProps) {
  return (
    <Menu>
      <MenuButton className="p-1 text-gray-300 hover:text-white" onClick={onClick}>
        <MoreVertical className="w-5 h-5" />
      </MenuButton>
      <MenuItems anchor="bottom end" className="bg-gray-900 border border-gray-700 rounded shadow-lg w-56 px-2 py-2 mt-4 z-50">
        {children}
      </MenuItems>
    </Menu>
  );
}
