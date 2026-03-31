import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Sidebar from '@/components/Sidebar';
import { ThemeProvider } from '@/components/ThemeProvider';
import ConfirmModal from '@/components/ConfirmModal';
import { Suspense } from 'react';
import AuthWrapper from '@/components/AuthWrapper';
import DocModal from '@/components/DocModal';
import os from 'os';

export const dynamic = 'force-dynamic';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Ostris - AI Toolkit',
  description: 'A toolkit for building AI things.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // Check if the AI_TOOLKIT_AUTH environment variable is set
  const authRequired = process.env.AI_TOOLKIT_AUTH ? true : false;

  const platform = os.platform();

  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="apple-mobile-web-app-title" content="AI-Toolkit" />
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                var theme = localStorage.getItem('theme') || 'dark';
                if (theme === 'dark') document.documentElement.classList.add('dark');
              })();
            `,
          }}
        />
      </head>
      <body className={inter.className}>
        <script dangerouslySetInnerHTML={{ __html: `window.server_platform = "${platform}";` }} />
        <ThemeProvider>
          <AuthWrapper authRequired={authRequired}>
            <div className="flex h-screen bg-gray-950">
              <Sidebar />
              <main className="flex-1 overflow-auto bg-gray-950 text-gray-100 relative">
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
