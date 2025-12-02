import { Card, CardContent } from "@/components/ui/card";
import { LucideIcon } from "lucide-react";

interface FeatureCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
}

export default function FeatureCard({ icon: Icon, title, description }: FeatureCardProps) {
  return (
    <Card className="hover-elevate" data-testid={`card-feature-${title}`}>
      <CardContent className="p-6 space-y-4">
        <div className="w-12 h-12 rounded-md bg-primary/10 flex items-center justify-center">
          <Icon className="h-6 w-6 text-primary" />
        </div>
        <h3 className="text-lg font-semibold" data-testid={`text-feature-title-${title}`}>
          {title}
        </h3>
        <p className="text-sm text-muted-foreground leading-relaxed" data-testid={`text-feature-desc-${title}`}>
          {description}
        </p>
      </CardContent>
    </Card>
  );
}
