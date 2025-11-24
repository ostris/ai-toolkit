import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
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
  webpack: (config) => {
    // Suppress osx-temperature-sensor warning on non-MacOS
    config.resolve.alias['osx-temperature-sensor'] = false;
    return config;
  },
};

export default nextConfig;
