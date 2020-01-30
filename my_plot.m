function my_plot(num_centroids)
    current_distance = 0;
    current_centroid = 0;
    points = csvread("input.csv");
    centroids = csvread("results.csv");
    colors = ['b' 'g' 'r' 'c' 'y' 'm' 'k'];
    figure
    view(3);
    box on
    hold on
    for a=1:2000
        for b=1:num_centroids
            if(norm(points(a,:) - centroids(b, :)) > current_distance)
                current_distance = norm(points(a,:) - centroids(b, :));
                current_centroid = b;
            end
        end
        scatter3(points(a, 1), points(a, 2), points(a, 3), 15, colors(current_centroid), 'MarkerFaceColor', colors(current_centroid));
        current_distance = 0;
    end
    
    for a=1:num_centroids
                scatter3(centroids(a, 1), centroids(a, 2), centroids(a, 3), 300, [0 0 0], 'MarkerFaceColor', [0 0 0]);
    end
end