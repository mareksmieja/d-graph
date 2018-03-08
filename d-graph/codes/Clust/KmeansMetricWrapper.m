function [centres, error, post, errlog] = KmeansMetricWrapper(centres, data, options)

    repeatTimes = 50;
    plotError = false;
    % random select k seeds
    options(5) = 1;
    options(2) = 1e-3;
    options(3) = 1e-3;
    
    if (plotError)
        errors = zeros(repeatTimes, 1);
    end
    % repeat 100 times and evaluate distortion
    error = realmax;
    for i=1:repeatTimes
        %fprintf('.');
        [centresP, optionsP, postP, errlogP] = kmeansMetric(centres, data, options);
        if ( optionsP(8) < error )
            centres  = centresP;
            error = optionsP(8);
            post = postP;
            errlog = errlogP;
        end
        if (plotError)
            errors(i) = optionsP(8);
        end
    end
    %fprintf('\n');
    if (plotError)
        plot(errors)
        drawnow
    end
end