interface CardProps {
  title?: string;
  children?: React.ReactNode;
}

const Card: React.FC<CardProps> = ({ title, children }) => {
  return (
    <section className="space-y-2 px-4 pb-4 pt-2 bg-gray-900 rounded-lg">
      {title && <h2 className="text-lg mb-2 font-semibold uppercase text-gray-500">{title}</h2>}
      {children ? children : null}
    </section>
  );
};

export default Card;
