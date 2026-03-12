import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Sidebar from '@/components/Sidebar';
import { ThemeProvider } from '@/components/ThemeProvider';
import ConfirmModal from '@/components/ConfirmModal';
import { Suspense } from 'react';
import AuthWrapper from '@/components/AuthWrapper';
import DocModal from '@/components/DocModal';

import { FaBars } from 'react-icons/fa6';

export const dynamic = 'force-dynamic';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Ostris - AI Toolkit',
  description: 'A toolkit for building AI things.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // Check if the AI_TOOLKIT_AUTH environment variable is set
  const authRequired = process.env.AI_TOOLKIT_AUTH ? true : false;

  return (
    <html lang="en" className="dark">
      <head>
        <meta name="apple-mobile-web-app-title" content="AI-Toolkit" />
      </head>
      <body className={inter.className}>
        <ThemeProvider>
          <AuthWrapper authRequired={authRequired}>
            <div className="flex h-screen bg-gray-950 group">
              <label htmlFor="sidebar-toggle" className="hidden fixed inset-0 z-30 bg-gray-900/75 group-has-[#sidebar-toggle:checked]:block">
              </label>
              <Sidebar className="fixed top-0 left-0 z-40 w-56 h-screen transition-transform -translate-x-full sm:translate-x-0 group-has-[#sidebar-toggle:checked]:translate-x-0" />
              <main className="flex-1 overflow-auto bg-gray-950 text-gray-100 relative sm:ml-56">
                <Suspense>{children}</Suspense>
              </main>
            </div>
          </AuthWrapper>
        </ThemeProvider>
        <ConfirmModal />
        <DocModal />
      </body>
    </html>
  );
}
