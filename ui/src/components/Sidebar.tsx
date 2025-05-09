import Link from 'next/link';
import { Home, Settings, BrainCircuit, Images, Plus } from 'lucide-react';

const Sidebar = () => {
  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: Home },
    { name: 'New Job', href: '/jobs/new', icon: Plus },
    { name: 'Training Jobs', href: '/jobs', icon: BrainCircuit },
    { name: 'Datasets', href: '/datasets', icon: Images },
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
      <a href="https://patreon.com/ostris" target="_blank" rel="noreferrer" className="flex items-center space-x-2 p-4">
        <div className="min-w-[26px] min-h-[26px]">
          <svg
            viewBox="0 0 512 512"
            xmlns="http://www.w3.org/2000/svg"
            fillRule="evenodd"
            clipRule="evenodd"
            strokeLinejoin="round"
            strokeMiterlimit="2"
          >
            <g transform="matrix(.47407 0 0 .47407 .383 .422)">
              <clipPath id="prefix__a">
                <path d="M0 0h1080v1080H0z"></path>
              </clipPath>
              <g clipPath="url(#prefix__a)">
                <path
                  d="M1033.05 324.45c-.19-137.9-107.59-250.92-233.6-291.7-156.48-50.64-362.86-43.3-512.28 27.2-181.1 85.46-237.99 272.66-240.11 459.36-1.74 153.5 13.58 557.79 241.62 560.67 169.44 2.15 194.67-216.18 273.07-321.33 55.78-74.81 127.6-95.94 216.01-117.82 151.95-37.61 255.51-157.53 255.29-316.38z"
                  fillRule="nonzero"
                  fill="#ffffff"
                ></path>
              </g>
            </g>
          </svg>
        </div>
        <div className="text-gray-500 text-md mb-2 flex-1 pt-2 pl-2">Support me on Patreon</div>
      </a>
    </div>
  );
};

export default Sidebar;
