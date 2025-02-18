import Link from 'next/link';
import { Home, Settings, BarChart2, BrainCircuit } from 'lucide-react';

const Sidebar = () => {
  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: Home },
    { name: 'Train', href: '/train', icon: BrainCircuit },
    { name: 'Settings', href: '/settings', icon: Settings },
  ];

  return (
    <div className="flex flex-col w-64 bg-gray-900 text-gray-100">
      <div className="p-4">
        <h1 className="text-xl">
            <img src="/ostris_logo.png" alt="Ostris AI Toolkit" className="w-auto h-8 mr-3 inline" />
            Ostris - AI Toolkit
        </h1>
      </div>
      <nav className="flex-1">
        <ul className="px-2 py-4 space-y-2">
          {navigation.map(item => (
            <li key={item.name}>
              <Link
                href={item.href}
                className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors"
              >
                <item.icon className="w-5 h-5 mr-3" />
                {item.name}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;
