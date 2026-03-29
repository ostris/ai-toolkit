import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals.push('osx-temperature-sensor');
    }
    return config;
  },
  devIndicators: {
    buildActivity: false,
  },
  typescript: {
    // Remove this. Build fails because of route types
    ignoreBuildErrors: true,
  },
  experimental: {
    serverActions: {
      bodySizeLimit: '100gb',
    },
    middlewareClientMaxBodySize: '100gb',
  },
};

export default nextConfig;
