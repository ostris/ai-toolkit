export default function Dashboard() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[1, 2, 3].map(i => (
          <div key={i} className="p-6 bg-gray-800 rounded-lg">
            <h2 className="text-xl font-semibold mb-2">Card {i}</h2>
            <p className="text-gray-400">Example dashboard card content</p>
          </div>
        ))}
      </div>
    </div>
  );
}
