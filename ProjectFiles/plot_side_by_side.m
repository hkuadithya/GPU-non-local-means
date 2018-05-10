for z=(50:120)
    subplot(1, 2, 1), imshow(ms_lesions(:, :, z), []);
    subplot(1, 2, 2), imshow(normal_brain(:, :, z), []);
    pause(1);
end;