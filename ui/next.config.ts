import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  basePath: '/aitoolkit',
  devIndicators: {
    buildActivity: false,
  },
  typescript: {
    // Remove this. Build fails because of route types
    ignoreBuildErrors: true,
  },
  experimental: {
    serverActions: {
      bodySizeLimit: '100mb',
    },
  },
};

export default nextConfig;
